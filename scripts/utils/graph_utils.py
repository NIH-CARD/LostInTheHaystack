import os
import re
import math
from pathlib import Path
from typing import Callable, Sequence, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter, LogLocator, FixedLocator, NullLocator
import seaborn as sns
from scipy.stats import bootstrap

SIZE_COLORS = {
    "sm_g": "#87d6ff",   # light blue
    "md_g": "#187bcd",   # medium blue
    "lg_g": "#03254c",   # dark blue
}


def boot_mean_ci(
    vals: pd.Series, confidence: float = 0.90, n_resamples: int = 10_000, method: str = "basic"
) -> tuple[float, float]:
    """
    Compute the mean and confidence interval half-width using bootstrap resampling.
    """
    arr = np.asarray(vals.dropna(), dtype=float)
    if arr.size == 0:
        return np.nan, np.nan

    res = bootstrap(
        (arr,),
        statistic=np.mean,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method=method,
        random_state=0
    )
    mean = arr.mean()
    half_width = mean - res.confidence_interval.low
    return mean, half_width


def _parse_stat(val) -> float:
    """Helper to strip the '± ci' part if present and return the point-estimate float."""
    if isinstance(val, (int, float)):
        return float(val)
    return float(str(val).split("±")[0])


def plot_heatmap(
    summary_df: pd.DataFrame, gold_sizes: list[str], bench: str, out_dir: Path, metric_label: str,
    vmin: float = 0.0, vmax: float = 1.0, baseline: bool = False
):
    """
    Plot a heatmap of performance metrics for different models and gold context sizes.
    """

    SM_FT = 15
    out_dir.mkdir(exist_ok=True)

    # Drop range column if present
    df = summary_df.drop(columns=[c for c in summary_df.columns if c == "range"], errors="ignore")

    # Parse numerical values and keep annotations
    def _parse_stat(val) -> float:
        return float(val) if isinstance(val, (int, float)) else float(str(val).split("±")[0])

    numeric = df.map(_parse_stat)
    annot = df.astype(str)

    fig, ax = plt.subplots(figsize=(22, max(6, len(df) * 1.05)))

    # Draw heatmap
    heat = sns.heatmap(
        numeric,
        annot=annot,
        fmt="",
        cmap="YlOrRd_r",
        vmin=vmin, vmax=vmax,
        annot_kws={"fontsize": SM_FT},
        cbar_kws={"label": metric_label, "pad": 0.025},
        linewidths=0.5,
        linecolor="white",
        ax=ax
    )

    # Colorbar font sizes
    cbar = heat.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(SM_FT + 3)
    cbar.ax.tick_params(labelsize=SM_FT)

    # Rename and format x-axis tick labels
    X_TICKS_RENAMING = {
        "avg": "Average", "no_ctx": "Closed Book", "sm_g": "Small Gold",
        "md_g": "Medium Gold", "lg_g": "Large Gold", "distractor": "Distractor",
        "distractor_5": "Distractor", "distractor_200": "Distractor", "baseline": "Gold Only"
    }
    xtick_labels = [X_TICKS_RENAMING.get(col, col) for col in df.columns]
    ax.set_xticklabels(xtick_labels, rotation=0, ha="center", fontsize=SM_FT)

    # Split model name and size
    pat = re.compile(r"(.+)_(sm_g|md_g|lg_g)$")
    model_to_idxs = {}
    for i, label in enumerate(df.index):
        m = pat.match(label)
        model = m.group(1) if m else label
        model_to_idxs.setdefault(model, []).append(i)

    # Y-axis tick labels
    size_map = {"sm_g": "small", "md_g": "medium", "lg_g": "large"}
    y_labels = [size_map.get(label[-4:], label) for label in df.index]
    y_pos = np.arange(len(y_labels)) + 0.5
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, va="center", fontsize=SM_FT)
    ax.tick_params(axis="y", length=0)

    if not baseline:
        # Space for model names on the left
        fig.subplots_adjust(left=0.35)

        # Draw model names along the y-axis
        for model, idxs in model_to_idxs.items():
            mid = idxs[len(idxs) // 2] + 0.5
            ax.text(
                x=-0.15, y=mid, s=model,
                transform=ax.get_yaxis_transform(),
                rotation=90, ha="center", va="center",
                fontsize=SM_FT, fontweight="bold"
            )

    # Draw separator lines between model groups
    for idxs in list(model_to_idxs.values())[:-1]:
        boundary = idxs[-1] + 1
        ax.hlines(y=boundary, xmin=0, xmax=df.shape[1], colors="black", linewidth=5)

    ax.set_ylabel("")
    title_ext = "baseline" if baseline else "main"

    fig.savefig(
        out_dir / f"{bench}_{title_ext}_heatmap.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close(fig)


def plot_spaghetti_gold_by_model(
    summary_df: pd.DataFrame, bench: str, sizes: Sequence[str], depths: Sequence[float], out_dir: Path, metric_label: str, max_depth: int | None = None,
):
    """
    Plot a line chart comparing model performance across gold context depths,
    for multiple gold sizes (e.g., small vs large gold).
    """

    SM_FT = 25
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select depth columns
    depth_cols = [str(d) for d in depths]
    if max_depth is not None:
        depth_cols = depth_cols[:max_depth]
    x_vals = depths[:len(depth_cols)]

    # Extract unique base model names (excluding size suffix)
    size_suffix = re.compile(r"_(?:sm_g|md_g|lg_g)$")
    base_models = sorted({
        size_suffix.sub("", idx) for idx in summary_df.index if size_suffix.search(idx)
    })
    base_models = [
        m for m in base_models
        if m not in {"LLaMA-3.1-8b", "GPT-4o-Mini", "Gemini-2.0-Flash-Lite"}
    ]

    # Assign colors and styles
    color_map = {
        model: col for model, col in zip(
            base_models, ["#164850", "#36BFB6", "#85FF85", "#FF5C5C"]
        )
    }
    style_map = {"lg_g": "-", "sm_g": "--"}

    # Create figure
    plt.figure(figsize=(9, 6.5))

    # Plot lines for each model-size pair
    for model in base_models:
        for sz in sizes:
            row_key = f"{model}_{sz}"
            if row_key not in summary_df.index:
                continue
            ys = [_parse_stat(summary_df.loc[row_key, col]) for col in depth_cols]
            plt.plot(
                x_vals,
                ys,
                linestyle=style_map.get(sz, "-."),
                linewidth=4,
                alpha=0.95,
                color=color_map[model],
                marker=""
            )

    # Axis formatting
    ax = plt.gca()
    plt.xlabel("Gold Position", fontsize=SM_FT)
    if "lg_g" in sizes:
        plt.ylabel(metric_label, fontsize=SM_FT)
        plt.yticks(np.arange(0.00, 1.01, 0.10), fontsize=SM_FT)
    else:
        ax.set_ylabel(None)
        ax.set_yticklabels([])

    plt.xticks(x_vals, fontsize=SM_FT)
    plt.ylim(0.29, 1.01)
    plt.xlim(-0.01, 1.01)
    plt.grid(True, linestyle="--", alpha=0.4)

    # Legend
    cur_models = []
    if "lg_g" in sizes:
        cur_models += base_models[:2]
    if "sm_g" in sizes:
        cur_models += base_models[2:]

    model_handles = [
        mpatches.Patch(color=color_map[m], label=m) for m in cur_models
    ]
    style_handles = []
    if "lg_g" in sizes:
        style_handles.append(mlines.Line2D([], [], color="black", linestyle="-", label="large gold"))
    if "sm_g" in sizes:
        style_handles.append(mlines.Line2D([], [], color="black", linestyle="--", label="small gold"))

    # Save figure
    fname = f"{bench}_spaghetti_gold_models_{'_'.join(sizes)}.png"
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_table_as_png(
    df: pd.DataFrame, title: str, save_dir: Path
) -> None:
    """
    Save a pandas DataFrame as a PNG image with an optional title.
    """

    # Create figure and disable axis
    fig, ax = plt.subplots(
        figsize=(max(6, df.shape[1] * 1.5), max(2, df.shape[0] * 0.4 + 1))
    )
    ax.axis("off")

    # Draw table
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.2)

    # Add title
    if title:
        ax.set_title(title, pad=12, fontsize=12, weight="bold")

    # Save figure
    fig.tight_layout()
    fig.savefig(save_dir / f"{title}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_line_by_pos(
    data: pd.DataFrame, metric: str, compute_fn: Callable[[pd.Series], Tuple[float, float]],  gold_sizes: Sequence[str],
    depths: Sequence[float], title: str, save_path: str, metric_label: str | None = None, show_legend: bool = False, 
    show_xaxis: bool = False,
):
    """
    Plot a line chart with error bars showing metric values across gold context positions
    for different gold sizes (small, medium, large).
    """

    SM_FT, BG_FT = 20, 28
    extra_h = int(show_legend) + int(show_xaxis)
    plt.figure(figsize=(10, 6 + extra_h))

    fallback_color = "#808080"  # default grey if color not found
    markers = ["o", "o", "o"]   # consistent marker style

    # Plot error bars for each gold size
    for size, marker in zip(gold_sizes, markers):
        means, errs = [], []
        for depth in depths:
            col = f"{size}@{depth}_{metric}"
            val, ci = compute_fn(data[col]) if col in data else (np.nan, np.nan)
            means.append(val)
            errs.append(ci)

        plt.errorbar(
            depths,
            means,
            yerr=errs,
            fmt=f"-{marker}",
            color=SIZE_COLORS.get(size, fallback_color),
            label=size,
            capsize=10,
            linewidth=5,
            alpha=0.9
        )

    # Legend formatting
    if show_legend:
        size_labels = {"sm_g": "small", "md_g": "medium", "lg_g": "large"}
        legend_handles = [
            mpatches.Patch(
                color=SIZE_COLORS.get(sz, fallback_color),
                label=size_labels.get(sz, sz)
            )
            for sz in gold_sizes
        ]
        plt.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.125),
            ncol=len(legend_handles),
            fontsize=SM_FT,
            frameon=False,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=1.2,
        )

    # X-axis formatting
    if show_xaxis:
        plt.xticks(depths, fontsize=SM_FT)
        plt.xlabel("Gold Position", fontsize=SM_FT)
    else:
        plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    # Y-axis formatting
    plt.ylabel(metric_label or metric.replace("_", " ").title(), fontsize=SM_FT)
    plt.yticks(np.arange(0.20, 1.01, 0.10), fontsize=SM_FT)
    plt.ylim(0.19, 1.01)
    plt.grid(True, which="both", alpha=0.5)

    # Save figure
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{title}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_bar(
    bench_metrics: dict[str, dict[str, tuple[float, float]]], bench: str, metric_label: str, out_dir: Path
) -> None:
    """
    Plot a grouped bar chart of model performance for different gold sizes,
    including confidence intervals.
    """
    SM_FT = 11
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract models and size categories
    models = list(bench_metrics.keys())
    sizes = list(next(iter(bench_metrics.values())).keys())

    # Construct DataFrames for mean values and confidence intervals
    data = {
        model: {size: bench_metrics[model][size][0] for size in sizes}
        for model in models
    }
    errors = {
        model: {size: bench_metrics[model][size][1] for size in sizes}
        for model in models
    }
    df = pd.DataFrame.from_dict(data, orient="index")[sizes]
    df_err = pd.DataFrame.from_dict(errors, orient="index")[sizes]

    # Assign colors
    bar_colors = [SIZE_COLORS.get(size, "#333333") for size in df.columns]

    # Create figure
    plt.figure(figsize=(max(10, len(models) * 4.0), 18))
    ax = df.plot(
        kind="bar",
        yerr=df_err.T.values,
        capsize=5,
        color=bar_colors,
        width=0.8,
        rot=30,
        legend=False
    )

    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_ylabel(metric_label, fontsize=SM_FT)
    ax.tick_params(axis="both", labelsize=SM_FT)

    # Custom legend
    size_labels = {"sm_g": "small", "md_g": "medium", "lg_g": "large"}
    legend_labels = [size_labels.get(size, size) for size in df.columns]
    legend_colors = [SIZE_COLORS.get(size, "#333333") for size in df.columns]
    patches = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(legend_colors, legend_labels)
    ]

    plt.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=len(patches),
        fontsize=SM_FT,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.2,
        frameon=False
    )

    # Axis limits and formatting
    plt.ylim(-0.01, 1.01)
    plt.yticks(np.arange(0.0, 1.01, 0.1), fontsize=SM_FT)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    # Save figure
    plt.savefig(out_dir / f"{bench}_bar.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_range_scatter(
    summary_df: pd.DataFrame, bench: str, gold_sizes: Sequence[str], out_dir: Path, metric_label: str,
) -> None:
    """
    Plot a scatter chart showing the range of performance metrics across models,
    grouped and color-coded by gold context size.
    """

    SM_FT = 12
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Helpers ---
    def _extract_size(idx: str, size_keys: Sequence[str]) -> str:
        for key in sorted(size_keys, key=len, reverse=True):
            if re.search(rf"(?:_|-){re.escape(key)}$", idx):
                return key
        return "unknown"

    def _to_float(val) -> float:
        return float(val) if isinstance(val, (int, float)) else float(str(val).split("±")[0])

    # --- Data Prep ---
    df = summary_df.copy()
    df["range_val"] = df["range"].map(_to_float)
    df["size"] = df.index.map(lambda x: _extract_size(x, gold_sizes))
    df["model"] = [
        re.sub(rf"(?:_|-){re.escape(sz)}$", "", idx) if sz != "unknown" else idx
        for idx, sz in zip(df.index, df["size"])
    ]
    df["x"] = np.arange(len(df))

    # --- Colors ---
    size_palette = {"sm_g": "#d0efff", "md_g": "#187bcd", "lg_g": "#03254c"}
    palette_iter = iter(sns.color_palette("pastel", n_colors=len(df)))
    df["color"] = [size_palette.get(sz, next(palette_iter)) for sz in df["size"]]

    # --- X-Ticks ---
    tick_labels = [model if i % 3 == 1 else "" for i, model in enumerate(df["model"])]
    tick_positions = df["x"].values

    # --- Plot ---
    plt.figure(figsize=(8, 6))

    # Lines connecting points from the same model
    for _, group in df.groupby("model"):
        if len(group) > 1:
            plt.plot(
                group["x"],
                group["range_val"],
                linewidth=2.0,
                color="black",
                alpha=0.75,
                zorder=1
            )

    # Scatter points
    plt.scatter(
        df["x"],
        df["range_val"],
        c=df["color"],
        edgecolors="black",
        s=150,
        zorder=2
    )

    # --- Axes ---
    plt.xticks(tick_positions, tick_labels, rotation=-30, ha="center", fontsize=SM_FT)
    plt.xlim(-0.5, len(df) - 0.5)
    plt.ylabel(f"{metric_label} Range", fontsize=SM_FT)
    plt.ylim(-0.01, 0.75)
    plt.yticks(np.arange(0.0, 0.75, 0.10), fontsize=SM_FT)
    plt.grid(True, which="both", alpha=0.5)

    # --- Legend ---
    size_labels = {"sm_g": "small", "md_g": "medium", "lg_g": "large"}
    legend_handles = [
        mpatches.Patch(color=size_palette[k], label=size_labels.get(k, k))
        for k in gold_sizes if k in size_palette
    ]
    plt.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=SM_FT,
        handlelength=1.5,
        handletextpad=0.6,
        columnspacing=1.2
    )

    # --- Save ---
    fname = f"{bench}_scatter_range_{metric_label.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_token_count_distribution(
    df: pd.DataFrame,
    gold_sizes: Sequence[str],
    out_dir: Path,
    bench: str,
    token_col_suffix: str = "token_count",
    title: str = "Token Count Distribution",
) -> None:
    """
    Plot histogram of token count distributions for different gold sizes,
    with symlog/linear scaling and median indicators.
    """

    SM_FT = 14
    LIN_THRESH = 500
    LIN_SCALE = 2
    LOG_THRESHOLD_RATIO = 50

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": SM_FT,
        "axes.labelsize": SM_FT,
        "xtick.labelsize": SM_FT,
        "ytick.labelsize": SM_FT,
    })
    sns.set_style("whitegrid")

    size_labels = {"sm_g": "small", "md_g": "medium", "lg_g": "large"}

    # Identify token columns
    token_cols: dict[str, str] = {}
    for size in gold_sizes:
        col = next(
            (c for c in df.columns if c.startswith(size) and c.endswith(token_col_suffix)),
            None
        ) or next(
            (c for c in df.columns if c.startswith(size) and "token" in c.lower()),
            None
        )
        if col is None:
            raise ValueError(
                f"No token-count column found for size '{size}'. "
                f"Available columns: {list(df.columns)}"
            )
        token_cols[size] = col

    # Trim outliers and compute medians
    filtered, medians = {}, {}
    global_lo, global_hi = [], []
    for size, col in token_cols.items():
        vals = df[col].dropna().to_numpy()
        if vals.size == 0:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        trimmed = vals[(vals >= lo) & (vals <= hi)]
        if bench == "cbb":
            trimmed = trimmed[trimmed <= 501]
        if trimmed.size == 0:
            trimmed = vals
        filtered[size] = trimmed
        medians[size] = int(np.median(vals))
        global_lo.append(trimmed.min())
        global_hi.append(trimmed.max())

    if not filtered:
        raise ValueError("No non-empty token distributions after trimming.")

    x_min, x_max_data = min(global_lo), max(global_hi)
    span_ratio = x_max_data / max(x_min, 1)

    # Choose binning strategy
    def _nice_step(lo: float, hi: float, target: int = 8) -> int:
        span = hi - lo
        raw = span / max(target, 1)
        mag = 10 ** math.floor(math.log10(raw))
        for m in (1, 2, 5, 10):
            step = m * mag
            if raw <= step:
                return int(step)
        return int(step)

    if span_ratio > LOG_THRESHOLD_RATIO:
        lin_step = max(10, int(np.ceil((LIN_THRESH - x_min) / 40 / 10) * 10))
        lin_bins = np.arange(x_min, LIN_THRESH, lin_step)
        log_bins = np.logspace(np.log10(LIN_THRESH), np.log10(x_max_data), 60)
        bins = np.unique(np.hstack([lin_bins, log_bins]))
    else:
        step = max(10, int(np.ceil((x_max_data - x_min) / 80 / 10) * 10))
        bins = np.arange(x_min, x_max_data + step, step)

    bins = bins[bins < x_max_data]
    bins = np.append(bins, x_max_data)

    # Plot
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ymax = 0.0

    def _k_format(x, _):
        return f"{x/1000:.0f}k".rstrip("0").rstrip(".") if x >= 1000 else str(int(x))

    if span_ratio > LOG_THRESHOLD_RATIO:
        ax.set_xscale("symlog", base=10, linthresh=LIN_THRESH, linscale=LIN_SCALE)
        lin_ticks = [0] + list(range(100, LIN_THRESH + 1, 100))
        log_locator = LogLocator(base=10, subs=(1.0, 2.0, 5.0))
        log_ticks = log_locator.tick_values(LIN_THRESH, x_max_data)
        ax.xaxis.set_major_locator(FixedLocator(lin_ticks + list(log_ticks)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0, x_max_data)
    else:
        if x_max_data < LIN_THRESH:
            ticks = np.arange(100, LIN_THRESH + 1, 100)
            ax.set_xlim(0, LIN_THRESH)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
        else:
            step = _nice_step(x_min, x_max_data, target=10)
            x_max_nice = math.ceil(x_max_data / step) * step
            ticks = np.arange(math.floor(x_min / step) * step, x_max_nice + step, step)
            ax.set_xlim(x_min, x_max_nice)
            ax.xaxis.set_major_locator(FixedLocator(ticks))

    ax.xaxis.set_major_formatter(FuncFormatter(_k_format))

    if bench == "cbb":
        ax.set_xlim(0, 500)
        ax.xaxis.set_major_locator(FixedLocator([100, 200, 300, 400, 500]))

    keep = [t for t in ax.get_xticks() if (t <= 0) or (t >= 100)]
    ax.xaxis.set_major_locator(FixedLocator(keep))
    ax.xaxis.set_minor_locator(NullLocator())

    for size in gold_sizes:
        data = filtered.get(size, np.array([]))
        if data.size == 0:
            continue
        color = SIZE_COLORS.get(size, "#999999")
        weights = np.ones_like(data) / data.size * 100
        n, _, _ = plt.hist(data, bins=bins, weights=weights, alpha=0.9,
                           edgecolor="black", color=color)
        ymax = max(ymax, n.max())
        plt.axvline(medians[size], linestyle="--", linewidth=3.0,
                    color=color, zorder=3)

    # Axes cosmetics
    plt.xlabel("Token Count")
    plt.ylabel("Frequency (%)")
    plt.ylim(0, ymax * 1.10)
    plt.yticks(np.arange(0, math.ceil((ymax + 5) / 5) * 5, 5))
    plt.grid(alpha=0.5)

    # Legend
    legend_handles = [
        mpatches.Patch(color=SIZE_COLORS[k], label=f"{size_labels.get(k, k)} ({medians[k]:,})")
        for k in gold_sizes if k in filtered
    ]
    plt.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=SM_FT,
        handlelength=1.5,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{title.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()