import argparse
import gzip
from pathlib import Path

import yaml

from scripts.models.llm_client import init_llm
from scripts.utils.utils import append_result, load_completed_task_ids, merge_dicts, print_, ungzip_file

# Benchmark name -> runner module
BENCHMARK_DISPATCH = {
    "cbb": scripts.utils.cbb_run,
    "nq": scripts.utils.nq_run,
    "nm": scripts.utils.nm_run,
}

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM experiments on a given benchmark"
    )
    parser.add_argument(
        "--exp-config", required=True,
        help="Path to a YAML that references benchmark and model configs"
    )
    args = parser.parse_args()
    print_(f"Benchmark Runner Initialized", fun="[*]")

    # Load and merge benchmark and model configs
    exp_path = Path(args.exp_config)
    exp_cfg = yaml.safe_load(exp_path.open())
    base_dir = exp_path.parent
    bench_cfg = yaml.safe_load((base_dir / exp_cfg["benchmark"]).open())
    model_cfg = yaml.safe_load((base_dir / exp_cfg["model"]).open())
    cfg = merge_dicts(bench_cfg, model_cfg)

    # Prepare output path
    bm = cfg["name"]
    mmdl = cfg["llm"]["model"].replace("/", "_")
    out_dir = Path("data") / "results" / bm / mmdl
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["output"] = {
        "path": str(out_dir / f"{bm}_{mmdl}_results.json")
    }

    # Initialize the main LLM client
    client_params = cfg["llm"].get("client_params", {})
    generation_params = {
        "provider": cfg["llm"]["provider"],
        "model": cfg["llm"]["model"],
        **cfg["llm"].get("gen_config", {})
    }
    llm, gen_config = init_llm(
        client_params=client_params,
        generation_params=generation_params
    )

    # Initialize grading LLM client if CBB benchmark
    if bm == "cbb":
        grading_client_params = cfg["grading_llm"].get("client_params", {})
        grading_gen_params = {
            "provider": cfg["grading_llm"]["provider"],
            "model": cfg["grading_llm"]["model"],
            **cfg["grading_llm"].get("gen_config", {})
        }
        grading_llm, grading_gen_config = init_llm(
            client_params=grading_client_params,
            generation_params=grading_gen_params
        )

    # Load tasks and previously completed task IDs
    utils = BENCHMARK_DISPATCH[bm]
    task_path = Path(cfg["tasks"]["path"])

    # If unzipped task file does not exist, unzip the .gz file
    if not task_path.exists() and task_path.with_suffix('.jsonl.gz').exists():
        print_(f"Original file {task_path} not found. Ungzipping...")
        ungzip_file(task_path.with_suffix('.jsonl.gz'), task_path)

    tasks = utils.load_tasks(task_path)
    completed = load_completed_task_ids(Path(cfg["output"]["path"]))

    # Extract dynamic parameters
    params = cfg["params"]
    distractor_sizes = params.get("distractor_sizes")
    depths = params.get("depths")
    max_tasks = params.get("n_tasks")

    print_(f"Running Benchmark `{bm.upper()}' on {len(tasks)} Tasks (up to {max_tasks}) with Model {mmdl} ({len(completed)} done)", fun="[*]")

    # Main experiment loop
    for task_id, task in enumerate(tasks):
        # if task_id in completed or task_id >= max_tasks:
        #     continue

        if bm == "cbb":
            result = utils.run_experiments_for_task(
                task, task_id, llm, gen_config, tasks,
                distractor_sizes, depths,
                grading_llm, grading_gen_config
            )
        else:
            result = utils.run_experiments_for_task(
                task, task_id, llm, gen_config, tasks,
                distractor_sizes, depths
            )

        # append_result(result, Path(cfg["output"]["path"]))
        print_(f"Task {task_id} complete.")
        break
    print_("Benchmark Runner Done", fun="[*]")

if __name__ == "__main__":
    main()