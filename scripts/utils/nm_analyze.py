from pathlib import Path
import json
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

from scripts.utils.graph_utils import (
    plot_line_by_pos,
    plot_bar,
    save_table_as_png,
    boot_mean_ci,
    plot_range_scatter,
    plot_token_count_distribution,
    plot_spaghetti_gold_by_model,
    plot_heatmap,
)
from scripts.utils.utils import print_

NAME_MAPPING = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-Mini",
    "gemini-2.0-flash": "Gemini-2.0-Flash",
    "gemini-2.0-flash-lite": "Gemini-2.0-Flash-Lite",
    "Meta-Llama-3.1-405B-Instruct": "LLaMA-3.1-405b",
    "Llama-3.3-70B-Instruct": "LLaMA-3.3-70b",
    "llama8b": "LLaMA-3.1-8b",
}


def load_single_model_to_df(
    filepath: Path, metric: str, size_map: Dict[str, str], depths: List[float], distractor_sizes: List[int]
) -> pd.DataFrame:
    """
    Load model result JSON into a dataframe.
    """
    data = json.loads(filepath.read_text(encoding="utf-8"))
    rows = []
    for record in data:
        row = {
            **record.get("gold_ctxs_meta", {}),
            **record.get("distractor_ctxs_meta", {}),
            f"no_ctx_{metric}": record["no_ctx"][metric],
            **{f"{s}_{metric}": record[a][metric] for s, a in size_map.items()},
            **{f"distractor_{k}_{metric}": record[f"distractor_{k}"][metric] for k in distractor_sizes},
            **{
                f"{s}@{d}_{metric}": record[f"{a}@{d:.1f}_{k}distractors"][metric]
                for s, a in size_map.items() for d in depths for k in distractor_sizes
            }
        }
        rows.append(row)
    return pd.DataFrame(rows)


def load_all_models_to_df(
    results_root: Path, bench_name: str, metric: str, size_map: Dict[str, str], depths: List[float], distractor_sizes: List[int]
) -> pd.DataFrame:
    """
    Load results from all models into a single dataframe.
    """
    frames = []
    for model_dir in sorted(results_root.iterdir()):
        model_key = model_dir.name
        df = load_single_model_to_df(
            model_dir / f"{bench_name}_{model_key}_results.json", metric, size_map, depths, distractor_sizes
        )
        df.insert(0, "model", NAME_MAPPING.get(model_key, model_key))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_stats_table_df(
    all_results: pd.DataFrame, metric: str, size_map: Dict[str, str], depths: List[float], compute_fn=boot_mean_ci, precision: int = 2
) -> pd.DataFrame:
    """
    Build a summary table of math-verify scores per model, size, and depth.
    """
    cols = [*map(str, depths), "avg", "range", "baseline"]
    rows = {}
    for model, gdf in all_results.groupby("model", sort=False):
        for s in size_map:
            values, depth_means = [], []
            for d in depths:
                mean, ci = compute_fn(gdf[f"{s}@{d}_{metric}"])
                values.append(f"{mean:.{precision}f}±{ci:.{precision}f}")
                depth_means.append(mean)
            avg, avg_ci = compute_fn(pd.Series(depth_means))
            rng = np.nanmax(depth_means) - np.nanmin(depth_means) if depth_means else np.nan
            base_mean, base_ci = compute_fn(gdf[f"{s}_{metric}"])
            values += [f"{avg:.{precision}f}±{avg_ci:.{precision}f}", f"{rng:.{precision}f}", f"{base_mean:.{precision}f}±{base_ci:.{precision}f}"]
            rows[f"{model}_{s}"] = values
    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    return df.reindex([f"{NAME_MAPPING[m]}_{s}" for m in NAME_MAPPING for s in size_map if f"{NAME_MAPPING[m]}_{s}" in df.index])


def build_baseline_table_df(
    all_results: pd.DataFrame, metric: str, size_map: Dict[str, str], distractor_sizes: List[int], compute_fn=boot_mean_ci, precision: int = 2
) -> pd.DataFrame:
    """
    Build baseline math-verify scores for distractor, no-context, and gold-only conditions.
    """
    cols = [*[f"distractor_{k}" for k in distractor_sizes], "no_ctx", *size_map.keys()]
    rows = {}
    for model, gdf in all_results.groupby("model", sort=False):
        cells = []
        for col in cols:
            mean, ci = compute_fn(gdf[f"{col}_{metric}"])
            cells.append(f"{mean:.{precision}f}±{ci:.{precision}f}")
        rows[model] = cells
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def build_bench_metrics_dict(
    all_results: pd.DataFrame, metric: str, size_map: Dict[str, str], depths: List[float], compute_fn=boot_mean_ci
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Construct metrics dictionary: model -> gold size -> (mean, CI)
    """
    bench_metrics = {}
    for model, gdf in all_results.groupby("model", sort=False):
        stats = {
            s: compute_fn(pd.Series(np.concatenate([
                gdf[f"{s}@{d}_{metric}"].dropna().to_numpy()
                for d in depths if f"{s}@{d}_{metric}" in gdf
            ]))) for s in size_map
        }
        bench_metrics[model] = stats
    ordered = {}
    for raw_name, display_name in NAME_MAPPING.items():
        if display_name in bench_metrics:
            ordered[display_name] = bench_metrics[display_name]
    return ordered


def run_model_analysis(model_config: dict, bench_config: dict) -> None:
    """
    Run analysis and plot results for a single model.
    """
    bench_name = bench_config.get("name", Path(bench_config["tasks"]["path"]).stem)
    model_id = model_config.get("llm", {}).get("model", "").replace("/", "_")
    result_file = Path("data/results") / bench_name / model_id / f"{bench_name}_{model_id}_results.json"
    output_dir = Path("data/images") / bench_name / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    params = bench_config["params"]
    metric = params["metric"]
    size_map = params["size_map"]
    depths = params["depths"]
    distractor_sizes = params["distractor_sizes"]

    for k in distractor_sizes if model_id == "gemini-2.0-flash" else distractor_sizes[:1]:
        df = load_single_model_to_df(result_file, metric, size_map, depths, [k])
        title = f"{model_id}_{bench_name.upper()}_{metric}_{k}distractors"
        plot_line_by_pos(
            data=df, metric=metric, compute_fn=boot_mean_ci, gold_sizes=list(size_map), depths=depths,
            title=title, save_path=str(output_dir), metric_label=metric,
            show_legend=(model_id == "gemini-2.0-flash"), show_xaxis=(model_id == "Llama-3.3-70B-Instruct")
        )

    print_(f"{bench_name} analysis on {model_id} complete → {output_dir}", "{+}")


def run_benchmark_analysis(bench_config: dict) -> None:
    """
    Run full benchmark analysis across all models.
    """
    bench_name = bench_config.get("name", Path(bench_config["tasks"]["path"]).stem)
    results_root = Path("data/results") / bench_name
    output_dir = Path("data/images") / bench_name
    output_dir.mkdir(parents=True, exist_ok=True)

    params = bench_config["params"]
    metric = params["metric"]
    size_map = params["size_map"]
    depths = params["depths"]
    distractor_sizes = params["distractor_sizes"][:1]

    all_results = load_all_models_to_df(results_root, bench_name, metric, size_map, depths, distractor_sizes)

    stats_df = build_stats_table_df(all_results, metric, size_map, depths)
    save_table_as_png(stats_df, f"{bench_name}_depth_stats", output_dir)
    plot_heatmap(stats_df, list(size_map), bench_name, output_dir, metric, vmin=0.1, vmax=1.0, baseline=False)

    baseline_df = build_baseline_table_df(all_results, metric, size_map, distractor_sizes)
    save_table_as_png(baseline_df, f"{bench_name}_baselines", output_dir)
    plot_heatmap(baseline_df, list(size_map), bench_name, output_dir, metric, vmin=0.0, vmax=1.0, baseline=True)

    metrics_dict = build_bench_metrics_dict(all_results, metric, size_map, depths)
    plot_bar(metrics_dict, bench_name, metric, output_dir)
    plot_range_scatter(stats_df, bench_name, list(size_map), output_dir, metric)
    plot_token_count_distribution(all_results, list(size_map), output_dir, bench_name, "token_count", f"{bench_name} Token Count Distribution")
    plot_spaghetti_gold_by_model(stats_df, bench_name, ["sm_g", "lg_g"], depths, output_dir, metric)

    print_(f"Benchmark-wide analysis complete for {bench_name} --> {output_dir}", "{+}")