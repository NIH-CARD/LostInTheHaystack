import string
from typing import List
import regex as re

from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig

def format_bioscore_prompt(question: str, gold_ans: str, pred_ans: str) -> (str, str):
    prompt = (
        "You are a highly knowledgeable and experienced expert in the healthcare and biomedical field, "
        "possessing extensive medical knowledge and practical expertise."
        "### Scoring Instructions for Evaluating Analyst Responses\n\n"
        "**Objective:** Evaluate an analyst's response against a gold standard.\n\n"
        "**Scoring Criteria:**\n"
        "- **Exact Match:** 3 points for an exact or equally accurate response.\n"
        "- **Close Match:** 2 points for a very close response with minor inaccuracies.\n"
        "- **Partial Match:** 1 point for a partially accurate response with significant omissions.\n"
        "- **Irrelevant Information (Harmless):** Deduct 0.5 points for harmless irrelevant information.\n"
        "- **Irrelevant Information (Distracting):** Deduct 1 point for distracting irrelevant information.\n"
        "- **No Match:** 0 points for no match.\n"
        "- **Not Knowing Response:** -1 point for stating lack of knowledge or abstaining. An example of this scenario is when Analyst Response says \'There are various studies, resources or databases on this topic that you can check ... but I do not have enough information on this topic.\'\n\n"
        "**Scoring Process:**\n"
        "1. **Maximum Score:** 3 points per question.\n"
        "2. **Calculate Score:** Apply criteria to evaluate the response.\n\n"
        f"**Question:** {question}\n"
        f"**Golden Answer:** {gold_ans}\n"
        f"**Analyst Response:** {pred_ans}\n"
        "## Your grading\n"
        "Using the scoring instructions above, grade the Analyst Response return only the numeric score on a scale from 0.0-3.0. If the response is stating lack of knowledge or abstaining, give it -1.0."
    )
    return prompt

def check_BioScore_response(response: str) -> tuple[float, bool]:
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", response)
    if match:
        number = float(match.group(0))
        if number in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            return number / 3.0, True
        if number == -1:
            return number, True
    return None, False

def grade_bioscore(question: str, gold_ans: str, pred_ans: str, llm, gen_config):
    prompt = format_bioscore_prompt(
        question=question, gold_ans=gold_ans, pred_ans=pred_ans
    )
    bioscore_response = llm.generate(prompt, gen_config)
    bioscore, valid = check_BioScore_response(bioscore_response)
    if valid:
        return bioscore
    else:
        print("!!!Invalid BioScore Response!!!")
        return -2

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    if not prediction:
        print("warning empty prediction")
        prediction = ""

    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

def math_verify_score(prediction: str, ground_truth: str, precision: int = 6) -> float:
    verify_func = math_metric(
        gold_extraction_target=(ExprExtractionConfig(),LatexExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=max,
        precision=precision
    )
    # Run verification on single-item lists
    try:
        grade, _ = verify_func(["\\boxed{" + f"{ground_truth}" + "}"], [prediction])
        return float(grade)
    except Exception as e:
        print(f"Error grading with math-verify, defaulting to 0.")
        return 0