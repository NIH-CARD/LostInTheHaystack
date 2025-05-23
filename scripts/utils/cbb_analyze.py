from pathlib import Path
import json
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from scipy.stats import bootstrap

from scripts.utils.graph_utils import (
    plot_line_by_pos,
    plot_bar,
    save_table_as_png,
    plot_range_scatter,
    plot_token_count_distribution,
    plot_spaghetti_gold_by_model,
    plot_heatmap
)
from scripts.utils.utils import print_


NAME_MAPPING = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-Mini",
    "gemini-2.0-flash": "Gemini-2.0-Flash",
    "gemini-2.0-flash-lite": "Gemini-2.0-Flash-Lite",
    "Meta-Llama-3.1-405B-Instruct": "LLaMA-3.1-405b",
    "Llama-3.3-70B-Instruct": "LLaMA-3.3-70b",
    "llama8b": "LLaMA-3.1-8b"
}


def compute_quality(vals: pd.Series, confidence: float = 0.90, n_resamples: int = 10_000) -> tuple[float, float]:
    """
    Compute quality rate (fraction bioscore ≥ 2/3) with bootstrapped confidence interval.
    """
    arr = vals.dropna().to_numpy()
    if arr.size == 0:
        return np.nan, np.nan

    # Binary array: 1 if val ≥ 2/3, else 0
    binary = (arr >= 2 / 3).astype(float)

    res = bootstrap(
        data=(binary,),
        statistic=np.mean,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method="basic",
        random_state=0
    )

    mean = binary.mean()
    half_width = mean - res.confidence_interval.low
    return mean, half_width


def load_single_model_to_df(
    filepath: Path, metric: str, gold_sizes: List[str], depths: List[float], agents: List[str]
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
            **{f"{s}_{metric}": record[s][metric] for s in gold_sizes},
            **{f"{a}_{metric}": record[f"{a}_doc"][metric] for a in agents},
            f"distractor_{metric}": record["distractor"][metric],
            **{
                f"{s}@{d}_{metric}": record[f"{s}@{d}"][metric]
                for s in gold_sizes for d in depths
            },
        }
        rows.append(row)
    return pd.DataFrame(rows)


def load_all_models_to_df(
    results_root: Path, bench_name: str, metric: str, gold_sizes: List[str], depths: List[float], agents: List[str]
) -> pd.DataFrame:
    """
    Load results from all models into a single dataframe.
    """
    frames = []
    for model_dir in sorted(results_root.iterdir()):
        model_key = model_dir.name
        df = load_single_model_to_df(
            model_dir / f"{bench_name}_{model_key}_results.json", metric, gold_sizes, depths, agents
        )
        df.insert(0, "model", NAME_MAPPING.get(model_key, model_key))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_quality_table_df(
    df_all: pd.DataFrame, metric: str, gold_sizes: List[str], depths: List[float], compute_fn=compute_quality, precision: int = 2
) -> pd.DataFrame:
    """
    Build a summary table of quality scores per model, size, and depth.
    """
    cols = [*map(str, depths), "avg", "range", "baseline"]
    rows = {}

    for model, gdf in df_all.groupby("model", sort=False):
        for size in gold_sizes:
            values, depth_means = [], []

            for d in depths:
                col = f"{size}@{d}_{metric}"
                mean, ci = compute_fn(gdf[col])
                values.append(f"{mean:.{precision}f}±{ci:.{precision}f}")
                depth_means.append(mean)

            pooled_vals = np.concatenate([
                gdf[f"{size}@{d}_{metric}"].dropna().to_numpy()
                for d in depths if f"{size}@{d}_{metric}" in gdf
            ])
            avg, avg_ci = compute_fn(pd.Series(pooled_vals))
            rng = np.nanmax(depth_means) - np.nanmin(depth_means) if depth_means else np.nan
            values += [f"{avg:.{precision}f}±{avg_ci:.{precision}f}", f"{rng:.{precision}f}"]

            base_mean, base_ci = compute_fn(gdf[f"{size}_{metric}"])
            values.append(f"{base_mean:.{precision}f}±{base_ci:.{precision}f}")

            rows[f"{model}_{size}"] = values

    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    return df.reindex([
        f"{NAME_MAPPING[m]}_{s}" for m in NAME_MAPPING for s in gold_sizes if f"{NAME_MAPPING[m]}_{s}" in df.index
    ])


def build_baseline_table_df(
    df_all: pd.DataFrame, metric: str, gold_sizes: List[str], compute_fn=compute_quality, precision: int = 2
) -> pd.DataFrame:
    """
    Build baseline quality scores for distractor, no-context, and gold-only conditions.
    """
    cols = ["distractor", "no_ctx", *gold_sizes]
    rows = {}
    for model, gdf in df_all.groupby("model", sort=False):
        cells = []
        for key in ["distractor", "no_ctx"] + gold_sizes:
            mean, ci = compute_fn(gdf[f"{key}_{metric}"])
            cells.append(f"{mean:.{precision}f}±{ci:.{precision}f}")
        rows[model] = cells

    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    return df.reindex([NAME_MAPPING[m] for m in NAME_MAPPING if NAME_MAPPING[m] in df.index])


def build_bench_metrics_dict(
    df_all: pd.DataFrame, metric: str, gold_sizes: List[str], depths: List[float], compute_fn=compute_quality
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Construct metrics dictionary: model -> gold size -> (mean, CI)
    """
    bench_metrics = {}
    for model, gdf in df_all.groupby("model", sort=False):
        size_stats = {
            size: compute_fn(pd.Series(np.concatenate([
                gdf[f"{size}@{d}_{metric}"].dropna().to_numpy()
                for d in depths if f"{size}@{d}_{metric}" in gdf
            ])))
            for size in gold_sizes
        }
        bench_metrics[model] = size_stats

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

    results_path = Path("data/results") / bench_name / model_id / f"{bench_name}_{model_id}_results.json"
    output_dir = Path("data/images") / bench_name / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    params = bench_config["params"]
    sizes = list(params["size_map"].keys())
    metric = params["metric"]
    agents = params["agents"]
    depths = params["depths"]

    df = load_single_model_to_df(results_path, metric, sizes, depths, agents)
    title = f"{model_id}_{bench_name.upper()}_QR"
    metric_label = "Quality Rate"

    plot_line_by_pos(
        data=df, metric=metric, compute_fn=compute_quality, gold_sizes=sizes, depths=depths,
        title=title, save_path=str(output_dir), metric_label=metric_label,
        show_legend=(model_id == "gemini-2.0-flash"), show_xaxis=(model_id == "Llama-3.3-70B-Instruct")
    )

    print_(f"{bench_name} analysis complete on {model_id} --> {output_dir}", fun="{+}")


def run_benchmark_analysis(bench_config: dict) -> None:
    """
    Run full benchmark analysis across all models.
    """
    bench_name = bench_config.get("name", Path(bench_config["tasks"]["path"]).stem)
    output_dir = Path("data/images") / bench_name
    output_dir.mkdir(parents=True, exist_ok=True)

    params = bench_config["params"]
    metric = params["metric"]
    gold_sizes = list(params["size_map"].keys())
    depths = params["depths"]
    agents = params["agents"]
    metric_label = "Quality Rate"

    df_all = load_all_models_to_df(Path("data/results") / bench_name, bench_name, metric, gold_sizes, depths, agents)

    summary_df = build_quality_table_df(df_all, metric, gold_sizes, depths)
    save_table_as_png(summary_df, f"{bench_name}_depth_stats", output_dir)
    plot_heatmap(summary_df, gold_sizes, bench_name, output_dir, metric_label, vmin=0.30, vmax=1.0, baseline=False)

    baseline_df = build_baseline_table_df(df_all, metric, gold_sizes)
    save_table_as_png(baseline_df, f"{bench_name}_baselines", output_dir)
    plot_heatmap(baseline_df, gold_sizes, bench_name, output_dir, metric_label, vmin=0.0, vmax=1.0, baseline=True)

    bench_metrics = build_bench_metrics_dict(df_all, metric, gold_sizes, depths)
    plot_bar(bench_metrics, bench_name, metric_label, output_dir)
    plot_range_scatter(summary_df, bench_name, gold_sizes, output_dir, metric_label)
    plot_token_count_distribution(df_all, gold_sizes, output_dir, bench_name, "token_count", f"{bench_name} Token Count Distribution")
    plot_spaghetti_gold_by_model(summary_df, bench_name, ["sm_g", "lg_g"], depths, output_dir, metric_label)
    plot_spaghetti_gold_by_model(summary_df, bench_name, ["sm_g"], depths[:], output_dir, metric_label)
    plot_spaghetti_gold_by_model(summary_df, bench_name, ["lg_g"], depths[:], output_dir, metric_label)

    print_(f"Benchmark-wide analysis complete for {bench_name} --> {output_dir}", fun="{+}")