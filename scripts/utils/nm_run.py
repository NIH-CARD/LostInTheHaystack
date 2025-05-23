import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tiktoken

from scripts.utils.metrics import math_verify_score
from scripts.utils.utils import count_tokens_tiktoken, load_jsonl, print_


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file containing tasks.
    """
    return load_jsonl(path)


def trim_tokens(text: str, n_tokens: int = 5000, model: str = "gpt-4o") -> str:
    """
    Trim a text string to the last `n_tokens` tokens using TikToken encoding.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    token_ids = encoding.encode(text)
    if len(token_ids) <= n_tokens:
        return text
    return encoding.decode(token_ids[-n_tokens:])


def get_gold_ctxs_varying_size(task: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Generate gold contexts of varying sizes and compute token metadata.
    """
    trimmed_generation = trim_tokens(task.get("generation"))

    variants = {
        "sm_g": f"The final answer to {task.get('question')} is {task.get('answer')}.",
        "md_g": f"The solution to {task.get('question')} is {task.get('solution')}, final answer is {task.get('answer')}.",
        "lg_g": f"The reasoning for {task.get('question')} is {trimmed_generation}, the solution is {task.get('solution')}, and the final answer is {task.get('answer')}.",
    }
    metas = {f"{k}_tokens": count_tokens_tiktoken(v) for k, v in variants.items()}
    return variants, metas


def get_distractor_ctxs(
    task: Dict[str, Any], all_tasks: List[Dict[str, Any]], distractor_sizes: List[int], n_tokens: int = 5000
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Sample and truncate distractor documents and compute token metadata.
    """
    all_distractors = [t for t in all_tasks if t != task]
    rng = random.Random(task.get("task_id"))
    sampled = rng.sample(all_distractors, k=max(distractor_sizes))

    truncated_ctxs = []
    for d in sampled:
        truncated = trim_tokens(d.get("generation", ""), n_tokens=n_tokens)
        d_copy = d.copy()
        d_copy["generation"] = truncated
        truncated_ctxs.append(d_copy)

    metas = {}
    for k in distractor_sizes:
        texts = [d["generation"] for d in truncated_ctxs[:k]]
        metas[f"distractor_{k}_tokens"] = sum(count_tokens_tiktoken(txt) for txt in texts)

    return truncated_ctxs, metas


def format_docs(docs: List[Dict[str, Any]]) -> str:
    """
    Format distractor documents into a single string.
    """
    return "\n\n".join(
        f"The solution to {d.get('question')} is {d.get('generation')}, final answer is {d.get('answer')}"
        for d in docs
    )


def format_prompt(question: str, docs: List[str]) -> str:
    """
    Format a prompt that includes distractor contexts.
    """
    prefix = (
        "Create an ANSWER to the QUESTION using only the provided DOCUMENTS (some of which might be irrelevant). "
        "Write nothing but your final answer in LaTeX within \\boxed{}. "
        "If you do not know the answer to a question, explicitly state so in \\boxed{I don't know}. "
    )
    prompt_lines = [f"QUESTION: {question}", "DOCUMENTS:"] + docs + [f"QUESTION: {question}", "ANSWER:"]
    return prefix + "\n" + "\n".join(prompt_lines)


def format_prompt_noctx(question: str) -> str:
    """
    Format a prompt with no context documents.
    """
    return (
        "Create an ANSWER to the QUESTION. "
        "Write nothing but your final answer in LaTeX within \\boxed{}. "
        "If you do not know the answer to a question, explicitly state so in \\boxed{I don't know}. "
        f"QUESTION: {question} ANSWER:"
    )


def aggregate(
    question: str, docs: List[str], gold_answer: str, llm: Any, gen_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate an answer with the LLM and score it using math_verify.
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

    mv_score = math_verify_score(answer, gold_answer)
    return {"answer": answer, "math-verify": mv_score}


def run_experiments_for_task(
    task: Dict[str, Any], task_id: int, llm: Any, gen_config: Any, all_tasks: List[Dict[str, Any]], 
    distractor_sizes: List[int], depths: List[float]
) -> Dict[str, Any]:
    """
    Run all experiments for a given task and return results.
    """
    question = task.get("question")
    gold_answer = task.get("answer")
    gold_ctxs, gold_meta = get_gold_ctxs_varying_size(task)
    distractors, distractor_meta = get_distractor_ctxs(task, all_tasks, distractor_sizes)

    results = {
        "task_id": task_id,
        "question": question,
        "answer": gold_answer,
        "gold_ctxs_meta": gold_meta,
        "distractor_ctxs_meta": distractor_meta,
    }

    # No context baseline
    results["no_ctx"] = aggregate(question, [], gold_answer, llm, gen_config)
    print_("finished no context baseline experiment.")

    # Gold-only context baselines
    for name, text in gold_ctxs.items():
        results[name] = aggregate(question, [text], gold_answer, llm, gen_config)
    print_("finished gold only baseline experiments.")

    # All distractors combined
    for k in distractor_sizes:
        docs = [format_docs([d]) for d in distractors[:k]]
        results[f"distractor_{k}"] = aggregate(question, docs, gold_answer, llm, gen_config)
    print_("finished distractor only baseline experiments.")

    # Mixed gold + distractors at varying depths
    for k in distractor_sizes:
        base_docs = [format_docs([d]) for d in distractors[:k]]
        for depth in depths:
            pos = int(depth * k)
            for name, text in gold_ctxs.items():
                docs = base_docs.copy()
                docs.insert(pos, text)
                key = f"{name}@{depth:.1f}_{k}distractors"
                results[key] = aggregate(question, docs, gold_answer, llm, gen_config)
        print_(f"finished varying size and depth with {k} distractors answers.")

    return results