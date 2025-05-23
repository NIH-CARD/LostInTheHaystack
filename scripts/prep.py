import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
import ir_datasets
import pandas as pd
import yaml

from scripts.utils.utils import load_jsonl, print_

def setup_caches(cache_dirs):
    """
    Set up cache directories for HF/IR datasets, etc.
    """
    for env, path in cache_dirs.items():
        os.environ[env] = path
        Path(path).mkdir(parents=True, exist_ok=True)

def prep_cbb(cfg):
    """
    Prepare CBB data from local CSV with agent retrieval data.
    """
    cbb_path = cfg['raw']['cbb_sql_path']
    output_path = Path(cfg["tasks"]["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print_(f"Loading CBB w/ SQL annotations file from {cbb_path!r}")
    cbb_data = pd.read_csv(Path(cbb_path))

    rows = []
    for _, rec in cbb_data.iterrows():
        rec_dict = rec.to_dict()
        # Drop unnecessary keys
        for key in ["general_query", "refined_query"]:
            rec_dict.pop(key, None)
        rows.append(rec_dict)
        if len(rows) == cfg["params"]["n_tasks"]:
            break

    with open(output_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print_(f"Wrote {len(rows)} tasks to {output_path}")

def prep_nq(cfg):
    """
    Prepare NQ benchmark using local HELMET distractor documents and ir_datasets Wikipedia context.
    """
    helmet_path = cfg['raw']['helmet_path']
    output_path = Path(cfg["tasks"]["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print_(f"Loading HELMET file from {helmet_path!r}")
    helmet_tasks = load_jsonl(Path(helmet_path))
    helmet_map = {rec["question"]: rec for rec in helmet_tasks}

    print_("Loading ir_datasets KILT")
    kilt = ir_datasets.load("kilt")
    docstore = kilt.docs_store()

    print_("Loading Hugging Face facebook/kilt_tasks nq")
    hf_nq = load_dataset("facebook/kilt_tasks", name="nq", split="validation")

    count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for ex in hf_nq:
            qid = ex["id"]
            question = ex["input"]

            if question not in helmet_map:
                continue

            outputs = ex.get("output", [])
            answers = [o["answer"] for o in outputs if len(o["answer"]) > 0]

            prov = next((o["provenance"][0] for o in outputs if o.get("provenance")), None)
            if not prov:
                continue

            wid = prov["wikipedia_id"]
            doc = docstore.get(wid)

            record = {
                "question": question,
                "answers": answers,
                "gold_doc": {
                    "title": doc.title,
                    "text_pieces": doc.text_pieces,
                },
                "metadata": {
                    "qid": qid,
                    "wid": wid,
                    "gold_idx": prov["start_paragraph_id"],
                    "start_char": prov["start_character"],
                    "end_char": prov["end_character"],
                },
                "distractor_ctxs": []
            }

            helmet = helmet_map[question]
            gold_texts = [ctx['text'] for ctx in helmet['positive_ctxs']]
            filtered = [c for c in helmet["ctxs"]
                        if c["text"] not in gold_texts and
                        not any(ans.lower() in c['text'].lower() for ans in answers)]

            num_distractors = max(cfg['params']['distractor_sizes'])
            record["distractor_ctxs"] = filtered[:num_distractors]

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count == cfg["params"]["n_tasks"]:
                break

    print_(f"Wrote {count} tasks to {output_path}")

def prep_nm(cfg):
    """
    Prepare NM dataset using the OpenR1Math220k version.
    """
    print_("Loading Hugging Face OpenR1-Math-220k")
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    rows = []

    for rec in ds:
        uuid = rec.get("uuid")
        problem = rec.get("problem")
        answer = rec.get("answer")
        solution = rec.get("solution")
        gens = rec.get("generations", [])
        reasoning_flags = rec.get("is_reasoning_complete", [])
        math_flags = rec.get("correctness_math_verify", [])

        for i, (reason_flag, math_flag) in enumerate(zip(reasoning_flags, math_flags)):
            if reason_flag and math_flag:
                rows.append({
                    "uuid": uuid,
                    "question": problem,
                    "answer": answer,
                    "solution": solution,
                    "generation": gens[i]
                })
                break

        if len(rows) == cfg["params"]["n_tasks"]:
            break

    output_path = Path(cfg["tasks"]["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print_(f"Wrote {len(rows)} tasks to {output_path}")

# Benchmark name -> preparation function
DISPATCH = {
    "cbb": prep_cbb,
    "nq": prep_nq,
    "nm": prep_nm,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config for this benchmark")
    args = parser.parse_args()
    print_("Prep Runner Initialized", fun="[*]")

    cfg = yaml.safe_load(open(args.config))
    setup_caches(cfg.get("cache_dirs", {}))
    name = cfg["name"]

    if name not in DISPATCH:
        raise ValueError(f"No prep function for '{name}'")

    print_(f"Preparing data for '{name}'", fun="-", count=5)
    DISPATCH[name](cfg)
    print_(f"Done preparing data for '{name}'", fun="-", count=5)
    
    print_("Prep Runner Done", fun="[*]")

if __name__ == "__main__":
    main()