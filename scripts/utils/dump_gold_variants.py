import csv, importlib, yaml
from pathlib import Path
from typing import Dict, Any


BENCHMARK_DISPATCH = {
    "cbb": "scripts.utils.cbb_run",
    "nq":  "scripts.utils.nq_run",
    "nm":  "scripts.utils.nm_run",
}


def load_benchmark_cfg(bench_cfg_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(bench_cfg_path.open())
    return {
        "bench_name": cfg["name"],
        "tasks_path": Path(cfg["tasks"]["path"]),
        "size_map":  cfg.get("params", {}).get("size_map", {
            "sm_g": "sm_g",
            "md_g": "md_g",
            "lg_g": "lg_g",
        }),
    }


def main(bench_config: str, out_dir: str = "data/analysis/gold_variants"):
    cfg = load_benchmark_cfg(Path(bench_config))
    utils = importlib.import_module(BENCHMARK_DISPATCH[cfg["bench_name"]])
    tasks = utils.load_tasks(cfg["tasks_path"])

    out_path = Path(out_dir) / f"{cfg['bench_name']}_gold_variants.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sm, md, lg = (cfg["size_map"][k] for k in ("sm_g", "md_g", "lg_g"))

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id", "uuid",
                sm, md, lg,
                f"{sm}_tokens", f"{md}_tokens", f"{lg}_tokens"
            ],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()

        for tid, t in enumerate(tasks):
            gold_ctxs, meta = utils.get_gold_ctxs_varying_size(t)
            writer.writerow({
                "task_id": tid,
                "uuid": t.get("uuid"),
                sm: gold_ctxs[sm],
                md: gold_ctxs[md],
                lg: gold_ctxs[lg],
                f"{sm}_tokens": meta[f"{sm}_tokens"],
                f"{md}_tokens": meta[f"{md}_tokens"],
                f"{lg}_tokens": meta[f"{lg}_tokens"],
            })
    print(f"[✓] Wrote {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Dump gold context variants to CSV")
    p.add_argument("--bench-config", required=True, help="Path to benchmarks/<name>.yaml")
    p.add_argument("--out-dir", default="data/analysis/gold_variants")
    main(**vars(p.parse_args()))
