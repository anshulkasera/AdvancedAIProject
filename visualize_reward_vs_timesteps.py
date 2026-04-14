#!/usr/bin/env python3
"""
Visualize Reward vs Timesteps from a JSON training results file.
Creates a single-panel plot with one curve per prompt style.

Usage:
  python visualize_reward_vs_timesteps.py presentation/hopper.json
  python visualize_reward_vs_timesteps.py presentation/hopper.json --smooth-window 6 --save-data
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STYLE_DISPLAY_NAMES = {
    "balanced_goal_first": "Balanced Goal First",
    "checklist_style": "Checklist Style",
    "minimal_stats_style": "Minimal Stats Style",
    "ppo_only_policy": "True Reward",
}

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
    },
    "dark": {
        "fig_bg": "#0f1722",
        "ax_bg": "#17212b",
        "text": "#e6edf3",
        "grid": "#3a4958",
    },
}


def display_style_name(style_key):
    return STYLE_DISPLAY_NAMES.get(style_key, str(style_key).replace("_", " ").title())


def apply_axis_theme(ax, theme_cfg):
    ax.set_facecolor(theme_cfg["ax_bg"])
    for spine in ax.spines.values():
        spine.set_color(theme_cfg["text"])
    ax.tick_params(colors=theme_cfg["text"])
    ax.xaxis.label.set_color(theme_cfg["text"])
    ax.yaxis.label.set_color(theme_cfg["text"])
    ax.title.set_color(theme_cfg["text"])


def load_json_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_data_points(data):
    data_points = data.get("data_points", [])
    by_style = {}
    for point in data_points:
        style = point.get("prompt_style")
        phase = point.get("phase", "warmup")
        by_style.setdefault(style, {}).setdefault(phase, []).append(point)
    return by_style


def get_all_phase_points(phases):
    points = []
    for phase_points in phases.values():
        points.extend(phase_points)
    return points


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


def create_visualization(
    json_file,
    output_dir="graphs",
    data_dir="data",
    save_data=False,
    normalize_mode="none",
    smooth_window=1,
    theme="light",
):
    print(f"Loading {json_file}...")
    data = load_json_data(json_file)

    by_style = parse_data_points(data)
    title_from_json = data.get("title", "Reward vs Timesteps")
    env_name = data.get("config", {}).get("env_name", "unknown").upper()

    theme_cfg = THEMES[theme]
    style_colors = DARK_STYLE_COLORS if theme == "dark" else STYLE_COLORS

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    fig.patch.set_facecolor(theme_cfg["fig_bg"])
    apply_axis_theme(ax, theme_cfg)

    for style, phases in by_style.items():
        all_points = get_all_phase_points(phases)
        all_points.sort(key=lambda x: x.get("timestep_end", 0))

        timesteps = [p.get("timestep_end", 0) for p in all_points]
        rewards = [p.get("avg_env_reward", 0) for p in all_points]
        rewards = normalize_series(rewards, mode=normalize_mode)
        rewards = smooth_series(rewards, window=smooth_window)

        ax.plot(
            timesteps,
            rewards,
            marker="o",
            linewidth=2.3,
            markersize=5.2,
            label=display_style_name(style),
            color=style_colors.get(style),
        )

    ylabel = "Average Environment Reward"
    if normalize_mode == "minmax":
        ylabel = "Normalized Reward (min-max)"
    elif normalize_mode == "zscore":
        ylabel = "Normalized Reward (z-score)"

    ax.set_xlabel("Timesteps", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(f"{title_from_json}\n{env_name} - Reward vs Timesteps", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.35, linestyle="--", color=theme_cfg["grid"])

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=10)

    fig.suptitle("")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_clean = env_name.lower().replace("/", "_")
    base_name = Path(json_file).stem

    # Intentionally omit hyperparameters from filename.
    output_stem = f"reward_vs_timesteps_{env_clean}_{base_name}_{timestamp}"
    output_filename = f"{output_dir}/{output_stem}.png"

    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot to: {output_filename}")

    if save_data:
        os.makedirs(data_dir, exist_ok=True)
        data_filename = f"{data_dir}/{output_stem}.json"
        with open(data_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved data to: {data_filename}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Reward vs Timesteps from a JSON file")
    parser.add_argument("json_file", type=str, help="Path to JSON file with training data")
    parser.add_argument("--output_dir", type=str, default="graphs", help="Directory to save output plots")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save archived JSON files")
    parser.add_argument("--save-data", "--save_data", action="store_true", help="Save same-named JSON copy")
    parser.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "minmax", "zscore"],
        help="Optional reward normalization mode",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average window for reward curve (>=1)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="light",
        choices=["light", "dark"],
        help="Plot theme",
    )

    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        raise SystemExit(f"Error: File not found: {args.json_file}")
    if args.smooth_window < 1:
        raise SystemExit("Error: --smooth-window must be >= 1")

    create_visualization(
        args.json_file,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        save_data=args.save_data,
        normalize_mode=args.normalize,
        smooth_window=args.smooth_window,
        theme=args.theme,
    )


if __name__ == "__main__":
    main()
