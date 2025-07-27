import csv, importlib, yaml
from pathlib import Path
from typing import Dict, Any, List

BENCHMARK_DISPATCH = {
    "cbb": "scripts.utils.cbb_run",
    "nq":  "scripts.utils.nq_run",
    "nm":  "scripts.utils.nm_run",
}

def load_benchmark_cfg(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.open())
    return {
        "bench": cfg["name"],
        "tasks": Path(cfg["tasks"]["path"]),
        "sizes": cfg.get("params", {}).get("size_map", {"sm_g": "sm_g", "md_g": "md_g", "lg_g": "lg_g"}),
        "distractor_sizes": cfg.get("params", {}).get("distractor_sizes", []),
    }

def build_distractor_string(utils, bench: str, task: Dict[str, Any], tasks: List[Dict[str, Any]], sizes: List[int]) -> str:
    if bench == "cbb":
        ctxs, _ = utils.get_distractor_ctxs(task)
        return utils.format_docs(ctxs)
    if bench == "nm":
        ctxs, _ = utils.get_distractor_ctxs(task, tasks, sizes)
        return utils.format_docs(ctxs)
    # "nq"
    ctxs, _ = utils.get_distractor_ctxs(task, sizes)
    return utils.format_docs(ctxs)

def dump_csv(bench_cfg: str, out_dir: str = "data/analysis/gold_variants") -> None:
    cfg   = load_benchmark_cfg(Path(bench_cfg))
    utils = importlib.import_module(BENCHMARK_DISPATCH[cfg["bench"]])
    tasks = utils.load_tasks(cfg["tasks"])

    out_path = Path(out_dir) / f"{cfg['bench']}_gold_variants.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sm, md, lg = (cfg["sizes"][k] for k in ("sm_g", "md_g", "lg_g"))
    headers = ["task_id", "uuid",
               sm, md, lg,
               f"{sm}_tokens", f"{md}_tokens", f"{lg}_tokens",
               "distractor_ctx"]

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for tid, task in enumerate(tasks):
            gold_ctxs, meta = utils.get_gold_ctxs_varying_size(task)
            distractor_str  = build_distractor_string(
                                  utils, cfg["bench"], task, tasks, cfg["distractor_sizes"])

            writer.writerow({
                "task_id":      tid,
                "uuid":         task.get("uuid"),
                sm:             gold_ctxs[sm],
                md:             gold_ctxs[md],
                lg:             gold_ctxs[lg],
                f"{sm}_tokens": meta[f"{sm}_tokens"],
                f"{md}_tokens": meta[f"{md}_tokens"],
                f"{lg}_tokens": meta[f"{lg}_tokens"],
                "distractor_ctx": distractor_str,
            })

    print(f"[✓] Wrote {out_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Dump gold context variants and distractors to CSV")
    p.add_argument("--bench-config", required=True, help="benchmarks/<name>.yaml")
    p.add_argument("--out-dir", default="data/analysis/gold_variants")
    args = p.parse_args()
    dump_csv(args.bench_config, args.out_dir)
