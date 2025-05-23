import argparse
import importlib
from pathlib import Path

import yaml

from scripts.utils.utils import print_

# Benchmark name -> analysis module
ANALYZE_DISPATCH = {
    "cbb": "scripts.utils.cbb_analyze",
    "nq": "scripts.utils.nq_analyze",
    "nm": "scripts.utils.nm_analyze",
}

def main():
    parser = argparse.ArgumentParser(
        description="Unified analysis for experiment-specific or benchmark-wide metrics"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--exp-config", "-c",
        help="Path to an experiment config YAML (with benchmark and model refs)"
    )
    group.add_argument(
        "--bench-config", "-b",
        help="Path to a benchmark config YAML (for benchmark-wide analysis)"
    )
    args = parser.parse_args()
    print_("Analysis Runner Initialized", fun="[*]")

    if args.exp_config: # --- Experiment analysis ---
        
        # Load experiment config
        exp_path = Path(args.exp_config)
        exp_cfg = yaml.safe_load(exp_path.open())
        base_dir = exp_path.parent

        # Load benchmark and model configs
        bench_cfg = yaml.safe_load((base_dir / exp_cfg["benchmark"]).open())
        model_cfg = yaml.safe_load((base_dir / exp_cfg["model"]).open())

        # Load analysis module
        bench = bench_cfg.get("name", (base_dir / exp_cfg["benchmark"]).stem)
        module_path = ANALYZE_DISPATCH.get(bench)
        analyze_mod = importlib.import_module(module_path)

        # Run analysis
        analyze_mod.run_model_analysis(
            model_config=model_cfg,
            bench_config=bench_cfg
        )

    else: # --- Benchmark-wide analysis ---
        
        # Load benchmark config
        bench_path = Path(args.bench_config)
        bench_cfg = yaml.safe_load(bench_path.open())

        # Load analysis module
        bench = bench_cfg.get("name", bench_path.stem)
        module_path = ANALYZE_DISPATCH.get(bench)
        analyze_mod = importlib.import_module(module_path)

        # Run analysis
        analyze_mod.run_benchmark_analysis(bench_config=bench_cfg)

    print_("Analysis Runner Done", fun="[*]")


if __name__ == "__main__":
    main()