#!/usr/bin/env python3
"""
Plot Reward vs Timesteps for exactly three JSON result files in one figure.

Usage:
  python visualize_reward_timesteps_three.py file1.json file2.json file3.json
  python visualize_reward_timesteps_three.py file1.json file2.json file3.json --smooth-window 3
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STYLE_COLORS = {
    "balanced_goal_first": "#1f77b4",
    "checklist_style": "#ff7f0e",
    "minimal_stats_style": "#2ca02c",
    "ppo_only_policy": "#d62728",
}

DARK_STYLE_COLORS = {
    "balanced_goal_first": "#6bb6ff",
    "checklist_style": "#ffb36b",
    "minimal_stats_style": "#77d68a",
    "ppo_only_policy": "#ff7f8a",
}

THEMES = {
    "light": {
        "fig_bg": "#ffffff",
        "ax_bg": "#ffffff",
        "text": "#1f1f1f",
        "grid": "#b0b0b0",
        "bar_positive": "#bfe8bf",
        "bar_negative": "#f4c2c2",
        "bar_edge": "#1f1f1f",
        "zero": "#333333",
    },
    "dark": {
        "fig_bg": "#0f1722",
        "ax_bg": "#17212b",
        "text": "#e6edf3",
        "grid": "#3a4958",
        "bar_positive": "#5a8f63",
        "bar_negative": "#8f5a62",
        "bar_edge": "#d0d7de",
        "zero": "#c9d1d9",
    },
}

STYLE_DISPLAY_NAMES = {
    "balanced_goal_first": "Balanced Goal First",
    "checklist_style": "Checklist Style",
    "minimal_stats_style": "Minimal Stats Style",
    "ppo_only_policy": "True Reward",
}


def display_style_name(style_key):
    return STYLE_DISPLAY_NAMES.get(style_key, style_key.replace("_", " ").title())


def apply_axis_theme(ax, theme_cfg):
    ax.set_facecolor(theme_cfg["ax_bg"])
    for spine in ax.spines.values():
        spine.set_color(theme_cfg["text"])
    ax.tick_params(colors=theme_cfg["text"])
    ax.xaxis.label.set_color(theme_cfg["text"])
    ax.yaxis.label.set_color(theme_cfg["text"])
    ax.title.set_color(theme_cfg["text"])


def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_data_points(data):
    data_points = data.get("data_points", [])
    by_style = {}
    for point in data_points:
        style = point.get("prompt_style", "unknown_style")
        phase = point.get("phase", "warmup")
        by_style.setdefault(style, {}).setdefault(phase, []).append(point)
    return by_style


def get_all_phase_points(phases):
    all_points = []
    for points in phases.values():
        all_points.extend(points)
    return all_points


def normalize_series(values, mode="none"):
    arr = np.asarray(values, dtype=float)
    if mode == "none" or arr.size == 0:
        return arr.tolist()

    if mode == "minmax":
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        denom = vmax - vmin
        if denom < 1e-12:
            return np.zeros_like(arr).tolist()
        return ((arr - vmin) / denom).tolist()

    if mode == "zscore":
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std < 1e-12:
            return np.zeros_like(arr).tolist()
        return ((arr - mean) / std).tolist()

    raise ValueError(f"Unsupported normalize mode: {mode}")


def smooth_series(values, window=1):
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size == 0:
        return arr.tolist()

    kernel = np.ones(int(window), dtype=float) / float(window)
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").tolist()


def summarize_best_vs_ppo(by_style):
    """Return best non-True-Reward style average and True Reward average."""
    style_means = {}
    for style, phases in by_style.items():
        all_points = get_all_phase_points(phases)
        rewards = [p.get("avg_env_reward") for p in all_points if p.get("avg_env_reward") is not None]
        if rewards:
            style_means[style] = float(np.mean(rewards))

    ppo_value = style_means.get("ppo_only_policy")

    non_ppo = {k: v for k, v in style_means.items() if k != "ppo_only_policy"}
    if non_ppo:
        best_style = max(non_ppo, key=non_ppo.get)
        best_value = non_ppo[best_style]
    else:
        best_style = None
        best_value = None

    return best_style, best_value, ppo_value


def plot_reward_vs_timesteps_three(json_files, output_dir, smooth_window=1, normalize_mode="none", theme="light"):
    theme_cfg = THEMES[theme]
    style_colors = DARK_STYLE_COLORS if theme == "dark" else STYLE_COLORS

    line_fig, line_axes = plt.subplots(1, 3, figsize=(20, 6))
    line_fig.patch.set_facecolor(theme_cfg["fig_bg"])
    for ax in line_axes:
        apply_axis_theme(ax, theme_cfg)

    compare_fig, compare_ax = plt.subplots(1, 1, figsize=(10, 7))
    compare_fig.patch.set_facecolor(theme_cfg["fig_bg"])
    apply_axis_theme(compare_ax, theme_cfg)

    comparisons = []

    for idx, json_file in enumerate(json_files):
        ax = line_axes[idx]
        file_name = Path(json_file).name
        env_name = Path(json_file).stem.upper()

        try:
            data = load_json_data(json_file)
        except (json.JSONDecodeError, OSError) as exc:
            ax.text(0.5, 0.5, f"Failed to load\n{file_name}\n{exc}", ha="center", va="center", color=theme_cfg["text"])
            ax.set_title(file_name, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            comparisons.append((env_name, None, None, None))
            continue

        by_style = parse_data_points(data)
        env_name = data.get("config", {}).get("env_name", Path(json_file).stem).upper()

        if not by_style:
            ax.text(0.5, 0.5, f"No data_points found\n{file_name}", ha="center", va="center", color=theme_cfg["text"])
            ax.set_title(f"{env_name} ({file_name})", fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            comparisons.append((env_name, None, None, None))
            continue

        for style, phases in by_style.items():
            all_points = get_all_phase_points(phases)
            all_points.sort(key=lambda x: x.get("timestep_end", 0))

            timesteps = [p.get("timestep_end", 0) for p in all_points]
            rewards = [p.get("avg_env_reward", 0) for p in all_points]
            rewards = normalize_series(rewards, mode=normalize_mode)
            rewards = smooth_series(rewards, window=smooth_window)

            color = style_colors.get(style)
            ax.plot(
                timesteps,
                rewards,
                marker="o",
                linewidth=2.2,
                markersize=4.8,
                label=display_style_name(style),
                color=color,
            )

        ax.set_title(f"{env_name} ({file_name})", fontweight="bold")
        ax.set_xlabel("Timesteps", fontweight="bold")
        if normalize_mode == "minmax":
            ylabel = "Normalized Reward (min-max)"
        elif normalize_mode == "zscore":
            ylabel = "Normalized Reward (z-score)"
        else:
            ylabel = "Average Environment Reward"
        if idx == 0:
            ax.set_ylabel(ylabel, fontweight="bold")

        ax.grid(True, alpha=0.35, linestyle="--", color=theme_cfg["grid"])
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=9)

        best_style, best_value, ppo_value = summarize_best_vs_ppo(by_style)
        comparisons.append((env_name, best_style, best_value, ppo_value))

    env_labels = []
    ratio_increases = []
    for env_name, best_style, best_value, ppo_value in comparisons:
        style_label = display_style_name(best_style) if best_style is not None else "No Prompt Style"
        env_labels.append(f"{env_name}\n({style_label})")
        if best_value is None or ppo_value is None or abs(ppo_value) < 1e-12:
            ratio_increases.append(np.nan)
        else:
            ratio_increases.append((best_value - ppo_value) / abs(ppo_value))

    x = np.arange(len(env_labels))
    bar_colors = [
        theme_cfg["bar_positive"] if np.isfinite(r) and r >= 0 else theme_cfg["bar_negative"]
        for r in ratio_increases
    ]
    compare_ax.bar(x, ratio_increases, width=0.55, color=bar_colors, edgecolor=theme_cfg["bar_edge"], linewidth=1.8)
    compare_ax.axhline(0.0, color=theme_cfg["zero"], linewidth=1.2, linestyle="--")

    for i, ratio in enumerate(ratio_increases):
        if np.isfinite(ratio):
            marker = "better" if ratio >= 0 else "worse"
            va = "bottom" if ratio >= 0 else "top"
            compare_ax.text(i, ratio, f"{ratio * 100:+.1f}% ({marker})", ha="center", va=va, fontsize=9)
        else:
            compare_ax.text(i, 0, "True Reward N/A", ha="center", va="bottom", fontsize=9, color=theme_cfg["text"])

    compare_ax.set_xticks(x)
    compare_ax.set_xticklabels(env_labels)
    compare_ax.set_ylabel("Ratio Increase vs True Reward", fontweight="bold")
    compare_ax.set_title("Best Style Improvement vs True Reward", fontweight="bold")
    compare_ax.grid(True, alpha=0.35, axis="y", linestyle="--", color=theme_cfg["grid"])

    line_fig.suptitle(
        f"Reward vs Timesteps Across 3 Files | Normalize: {normalize_mode} | Smooth: {smooth_window}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
        color=theme_cfg["text"],
    )
    line_fig.tight_layout(rect=[0, 0, 1, 0.93])

    compare_fig.suptitle(
        "Best Style Improvement vs True Reward",
        fontsize=13,
        fontweight="bold",
        y=0.98,
        color=theme_cfg["text"],
    )
    compare_fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines_output_path = os.path.join(output_dir, f"reward_vs_timesteps_three_{ts}.png")
    compare_output_path = os.path.join(output_dir, f"reward_comparison_bar_{ts}.png")

    line_fig.savefig(lines_output_path, dpi=150, bbox_inches="tight")
    compare_fig.savefig(compare_output_path, dpi=150, bbox_inches="tight")
    plt.close(line_fig)
    plt.close(compare_fig)

    return lines_output_path, compare_output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot Reward vs Timesteps from exactly 3 JSON files and a separate comparison bar chart"
    )
    parser.add_argument("json_files", nargs=3, help="Exactly three JSON input files")
    parser.add_argument("--output-dir", default="graphs", help="Where to save output plot")
    parser.add_argument(
        "--normalize",
        default="none",
        choices=["none", "minmax", "zscore"],
        help="Optional reward normalization",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average smoothing window (>=1)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="light",
        choices=["light", "dark"],
        help="Plot theme",
    )

    args = parser.parse_args()

    if args.smooth_window < 1:
        raise SystemExit("Error: --smooth-window must be >= 1")

    missing = [p for p in args.json_files if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Error: Missing file(s): {', '.join(missing)}")

    lines_path, compare_path = plot_reward_vs_timesteps_three(
        args.json_files,
        output_dir=args.output_dir,
        smooth_window=args.smooth_window,
        normalize_mode=args.normalize,
        theme=args.theme,
    )

    print(f"Saved line plot image to: {lines_path}")
    print(f"Saved comparison bar image to: {compare_path}")


if __name__ == "__main__":
    main()
