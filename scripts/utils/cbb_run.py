import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.utils.metrics import grade_bioscore
from scripts.utils.utils import count_tokens_tiktoken, load_jsonl, print_

# Agent names used for distractor documents
AGENTS = ["Agent_V", "Agent_N", "Agent_T", "Agent_G"]


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file containing tasks.
    """
    return load_jsonl(path)


def get_gold_ctxs_varying_size(task: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Generate gold contexts of varying sizes and compute token metadata.
    """
    answer = task.get("answer")
    sql_gen = task.get("benchmark_query")
    returned_rows = task.get("execution_results")

    variants = {
        "sm_g": f"{answer}",
        "md_g": f"{sql_gen} {answer}",
        "lg_g": f"{sql_gen} {returned_rows} {answer}",
    }

    metas = {f"{k}_tokens": count_tokens_tiktoken(v) for k, v in variants.items()}
    return variants, metas


def get_distractor_ctxs(task: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Extract distractor contexts from the task and compute token metadata.
    """
    ctxs = {}
    metas = {}

    for agent in AGENTS:
        ctx = task.get(f"{agent}_Document")
        if ctx is None or (isinstance(ctx, float) and str(ctx) == "nan"):
            ctx = ""
        elif not isinstance(ctx, str):
            ctx = str(ctx)

        ctxs[f"{agent}_doc"] = ctx

        try:
            metas[f"{agent}_tokens"] = count_tokens_tiktoken(ctx)
        except Exception as e:
            print(f"[!] Error tokenizing {agent}: {e} | raw ctx: {ctx!r}")
            metas[f"{agent}_tokens"] = 0

    metas["distractor_tokens"] = sum(metas[f"{agent}_tokens"] for agent in AGENTS)
    return ctxs, metas


def format_docs(docs: List[Dict[str, Any]]) -> str:
    """
    Format a list of documents as concatenated text.
    """
    return "\n\n".join(f"Document: {val}" for _, val in docs.items())


def format_prompt(question: str, docs: List[str]) -> str:
    """
    Format a prompt that includes context documents.
    """
    prefix = (
        "You are a highly knowledgeable and experienced expert in the healthcare and biomedical field, "
        "possessing extensive medical knowledge and practical expertise. "
        "Create an answer to the question using only the provided documents (some of which might be irrelevant). "
        "If you cannot answer the question based on the documents, explicitly state that you do not know. "
    )

    prompt_lines = [f"Question: {question}", "Documents:"] + docs
    return prefix + "\n".join(prompt_lines)


def format_prompt_noctx(question: str) -> str:
    """
    Format a prompt without any context documents.
    """
    return (
        "You are a highly knowledgeable and experienced expert in the healthcare and biomedical field, "
        "possessing extensive medical knowledge and practical expertise. "
        "If you do not know the answer to a question, explicitly state that you do not know. "
        "If you do know the answer give the answer first, then give any supporting evidence."
        f"Question: {question}."
    )


def aggregate(
    question: str, docs: List[str], gold_answer: str, llm: Any, gen_config: Any, grading_llm: Any, grading_gen_config: Any,
) -> Dict[str, Any]:
    """
    Run LLM inference and grade the response using BioScore.
    """
    prompt = format_prompt_noctx(question) if len(docs) == 0 else format_prompt(question, docs)

    for attempt in range(3):
        try:
            answer = llm.generate(prompt, gen_config)
            break
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}): {e}")
            time.sleep((attempt + 1) * 3)
    else:
        print("LLM.generate failed 3 times")
        answer = ""

    bioscore = grade_bioscore(question, gold_answer, answer, grading_llm, grading_gen_config)
    return {"answer": answer, "bioscore": bioscore}


def run_experiments_for_task(
    task: Dict[str, Any], task_id: int, llm: Any, gen_config: Any, all_tasks: List[Dict[str, Any]], 
    distractor_sizes: List[int], depths: List[float], grading_llm: Any, grading_gen_config: Any,
) -> Dict[str, Any]:
    """
    Run a suite of experiments for a single task and collect results.
    """
    uuid = task.get("uuid")
    template_uuid = task.get("template_uuid")
    question = task.get("question")
    gold_answer = task.get("answer")
    bio_category = task.get("Bio_Category")
    sql_category = task.get("SQL_Category")

    gold_ctxs, gold_meta = get_gold_ctxs_varying_size(task)
    distractors, distractor_meta = get_distractor_ctxs(task)

    results = {
        "task_id": task_id,
        "cbb_uuid": uuid,
        "cbb_template_uuid": template_uuid,
        "question": question,
        "gold_answer": gold_answer,
        "bio_category": bio_category,
        "sql_category": sql_category,
        "gold_ctxs_meta": gold_meta,
        "distractor_ctxs_meta": distractor_meta,
    }

    # No context baseline
    results["no_ctx"] = aggregate(question, [], gold_answer, llm, gen_config, grading_llm, grading_gen_config)
    print_("finished no context baseline experiment.")

    # Gold-only context baselines
    for name, text in gold_ctxs.items():
        results[name] = aggregate(question, [text], gold_answer, llm, gen_config, grading_llm, grading_gen_config)
    print_("finished gold only baseline experiments.")

    # Individual distractor-only baselines
    for name, text in distractors.items():
        results[name] = aggregate(question, [text], gold_answer, llm, gen_config, grading_llm, grading_gen_config)
    print_("finished distractor-agent only baseline experiments.")

    # All distractors combined
    results["distractor"] = aggregate(
        question, [format_docs(distractors)], gold_answer, llm, gen_config, grading_llm, grading_gen_config
    )
    print_("finished all distractor-agent baseline experiments.")

    # Mixed gold + distractors at varying depths
    n_agents = len(AGENTS)
    distractor_list = list(distractors.values())

    for gold_name, gold_text in gold_ctxs.items():
        for depth in range(n_agents + 1):
            cur_ctx = distractor_list[:depth] + [gold_text] + distractor_list[depth:]
            results[f"{gold_name}@{depth / n_agents}"] = aggregate(
                question, cur_ctx, gold_answer, llm, gen_config, grading_llm, grading_gen_config
            )
    print_("finished varying size and depth with all agent distractors answers.")

    return results