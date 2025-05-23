import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.utils.metrics import best_subspan_em
from scripts.utils.utils import count_tokens_tiktoken, load_jsonl, print_

def load_tasks(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file containing tasks.
    """
    return load_jsonl(path)

def _resolve_snippet_bounds(paragraph: str, answer: str, start: int, end: int) -> Tuple[int, int]:
    """
    Resolve snippet bounds using either provided indices or by searching for the answer string.
    """
    if start == -1 or end == -1:
        idx = paragraph.find(answer)
        if idx != -1:
            return idx, idx + len(answer)
        return 0, min(len(paragraph), len(answer))
    return start, end

def _extract_sentence(paragraph: str, start: int, end: int, delims: str = ".!?") -> str:
    """
    Extract the sentence containing the answer span.
    """
    left_idx = max((paragraph.rfind(d, 0, start) for d in delims), default=-1)
    left = 0 if left_idx == -1 else left_idx + 1
    right_positions = [pos for pos in (paragraph.find(d, end) for d in delims) if pos != -1]
    right = (min(right_positions) + 1) if right_positions else len(paragraph)
    return paragraph[left:right].strip()

def get_gold_ctxs_varying_size(task: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Construct gold context variants of different sizes and their token counts.
    """
    gold_doc = task['gold_doc']
    meta = task['metadata']
    title = gold_doc['title']
    pieces = gold_doc['text_pieces']
    idx = int(meta['gold_idx'])
    paragraph = pieces[idx]
    article = "\n".join(pieces)

    raw_start = int(meta.get('start_char', -1))
    raw_end = int(meta.get('end_char', -1))
    answer = task.get('answers', [''])[0]
    start_char, end_char = _resolve_snippet_bounds(paragraph, answer, raw_start, raw_end)

    sentence = _extract_sentence(paragraph, start_char, end_char)
    section_start = max(0, idx - 4)
    section = "\n".join(pieces[section_start: section_start + 9])

    variants = {
        'gold_sentence':  f"{title} {sentence}",
        'gold_paragraph': f"{title} {paragraph}",
        'gold_section':   f"{title} {section}",
        'gold_article':   f"{title} {article}",
    }

    metas = {f"{k}_tokens": count_tokens_tiktoken(v) for k, v in variants.items()}
    return variants, metas

def get_distractor_ctxs(task: Dict[str, Any], distractor_sizes: List[int]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Return distractor contexts and token counts for different sizes.
    """
    ctxs = task.get('distractor_ctxs', [])
    metas = {}
    for k in distractor_sizes:
        texts = [format_docs([ctx]) for ctx in ctxs[:k]]
        metas[f"distractor_{k}_tokens"] = sum(count_tokens_tiktoken(txt) for txt in texts)
    return ctxs, metas

def format_docs(docs: List[Dict[str, Any]]) -> str:
    """
    Format a list of documents as concatenated strings.
    """
    return "\n\n".join(
        f"Title: {d.get('title', '')}\nDocument: {d.get('text', '')}" for d in docs
    )

def format_prompt(question: str, docs: List[str]) -> str:
    """
    Format a prompt including context documents.
    """
    prefix = (
        "Create an answer to the question using only the provided documents (some of which might be irrelevant). "
        "If you cannot answer the question based on the documents, explicitly state that you do not know. "
    )
    prompt_lines = [f"Question: {question}", "Documents:"] + docs
    return prefix + "\n".join(prompt_lines)

def format_prompt_noctx(question: str) -> str:
    """
    Format a prompt without any context.
    """
    return (
        "If you do not know the answer to a question, explicitly state that you do not know. "
        "If you do know the answer give the answer first, then give any supporting evidence."
        f"Question: {question}."
    )

def aggregate(question: str, docs: List[str], gold_answers: List[str], llm: Any, gen_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run LLM inference and compute subspan EM.
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

    subEM = best_subspan_em(answer, gold_answers)
    return {"answer": answer, "subEM": subEM}

def run_experiments_for_task(
    task: Dict[str, Any], task_id: int, llm: Any, gen_config: Any,
    all_tasks: List[Dict[str, Any]], distractor_sizes: List[int], depths: List[float]
) -> Dict[str, Any]:
    """
    Run a suite of LLM experiments on a single task.
    """
    question = task.get('question', '')
    gold_answers = task.get('answers', [])
    gold_ctxs, gold_meta = get_gold_ctxs_varying_size(task)
    distractors, distractor_meta = get_distractor_ctxs(task, distractor_sizes)

    results = {
        'task_id': task_id,
        'question': question,
        'gold_answers': gold_answers,
        'gold_ctxs_meta': gold_meta,
        'distractor_ctxs_meta': distractor_meta,
    }

    # No context baseline
    results['no_ctx'] = aggregate(question, [], gold_answers, llm, gen_config)
    print_("finished no context baseline experiment.")

    # Gold-only context baselines
    for name, text in gold_ctxs.items():
        results[name] = aggregate(question, [text], gold_answers, llm, gen_config)
    print_("finished gold only baseline experiments.")

    # All distractors combined baselines
    for k in distractor_sizes:
        docs = [format_docs([d]) for d in distractors[:k]]
        results[f'distractor_{k}'] = aggregate(question, docs, gold_answers, llm, gen_config)
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
                results[key] = aggregate(question, docs, gold_answers, llm, gen_config)
        print_(f"finished varying size and depth with {k} distractors answers.")

    return results