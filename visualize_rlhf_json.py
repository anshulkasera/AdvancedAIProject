#!/usr/bin/env python3
"""
Visualize RLLLM training results from JSON output files.
Creates a 4-panel plot similar to 4_gae_pg_gymnasium.py with hyperparameters embedded.

Usage:
    python visualize_rlhf_json.py pendulem2.json
    python visualize_rlhf_json.py hopper_results.json --output_dir graphs
"""

import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


STYLE_DISPLAY_NAMES = {
    'balanced_goal_first': 'Balanced Goal First',
    'checklist_style': 'Checklist Style',
    'minimal_stats_style': 'Minimal Stats Style',
    'ppo_only_policy': 'True Reward',
}

LIGHT_STYLE_COLORS = {
    'balanced_goal_first': '#1f77b4',
    'checklist_style': '#ff7f0e',
    'minimal_stats_style': '#2ca02c',
    'ppo_only_policy': '#d62728',
}

DARK_STYLE_COLORS = {
    'balanced_goal_first': '#6bb6ff',
    'checklist_style': '#ffb36b',
    'minimal_stats_style': '#77d68a',
    'ppo_only_policy': '#ff7f8a',
}

THEMES = {
    'light': {
        'fig_bg': '#ffffff',
        'ax_bg': '#ffffff',
        'text': '#1f1f1f',
        'grid': '#b0b0b0',
        'warmup_fill': '#f4c2c2',
        'warmup_edge': '#d62728',
        'rlllm_fill': '#cce5ff',
        'rlllm_edge': '#1f77b4',
        'true_fill': '#f4c2c2',
        'true_edge': '#d62728',
    },
    'dark': {
        'fig_bg': '#0f1722',
        'ax_bg': '#17212b',
        'text': '#e6edf3',
        'grid': '#3a4958',
        'warmup_fill': '#8f5a62',
        'warmup_edge': '#ff7f8a',
        'rlllm_fill': '#395f86',
        'rlllm_edge': '#6bb6ff',
        'true_fill': '#8f5a62',
        'true_edge': '#ff7f8a',
    },
}


def display_style_name(style_key):
    """Return a human-readable label for a prompt style key."""
    return STYLE_DISPLAY_NAMES.get(style_key, str(style_key).replace('_', ' ').title())


def apply_axis_theme(ax, theme_cfg):
    ax.set_facecolor(theme_cfg['ax_bg'])
    for spine in ax.spines.values():
        spine.set_color(theme_cfg['text'])
    ax.tick_params(colors=theme_cfg['text'])
    ax.xaxis.label.set_color(theme_cfg['text'])
    ax.yaxis.label.set_color(theme_cfg['text'])
    ax.title.set_color(theme_cfg['text'])


def load_json_data(filename):
    """Load JSON file and extract config, data points."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def parse_data_points(data):
    """Convert data_points list into structured format by style and phase."""
    data_points = data.get('data_points', [])
    
    # Organize by style
    by_style = {}
    for point in data_points:
        style = point.get('prompt_style')
        if style not in by_style:
            by_style[style] = {}
        phase = point.get('phase', 'warmup')
        if phase not in by_style[style]:
            by_style[style][phase] = []
        by_style[style][phase].append(point)
    
    return by_style


def get_all_phase_points(phases):
    """Flatten all phase lists for one prompt style."""
    all_points = []
    for pts in phases.values():
        all_points.extend(pts)
    return all_points


def normalize_series(values, mode='none'):
    """Normalize a numeric series using min-max or z-score scaling."""
    arr = np.asarray(values, dtype=float)
    if mode == 'none' or arr.size == 0:
        return arr.tolist()

    if mode == 'minmax':
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        denom = vmax - vmin
        if denom < 1e-12:
            return np.zeros_like(arr).tolist()
        return ((arr - vmin) / denom).tolist()

    if mode == 'zscore':
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std < 1e-12:
            return np.zeros_like(arr).tolist()
        return ((arr - mean) / std).tolist()

    raise ValueError(f"Unsupported normalize mode: {mode}")


def smooth_series(values, window=1):
    """Apply moving-average smoothing; window=1 leaves data unchanged."""
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size == 0:
        return arr.tolist()

    kernel = np.ones(int(window), dtype=float) / float(window)
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid').tolist()


def create_visualization(
    json_file,
    output_dir='graphs',
    data_dir='data',
    save_data=False,
    normalize_mode='none',
    smooth_window=1,
    theme='light',
):
    """Create 4-panel visualization from JSON training data."""
    
    # Load data
    print(f"Loading {json_file}...")
    data = load_json_data(json_file)
    
    config = data.get('config', {})
    details = data.get('details', {})
    title_from_json = data.get('title', 'RLLLM Training Results')
    
    theme_cfg = THEMES[theme]
    colors = DARK_STYLE_COLORS if theme == 'dark' else LIGHT_STYLE_COLORS

    # Parse data points by style
    by_style = parse_data_points(data)
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(theme_cfg['fig_bg'])
    for row in axes:
        for axis in row:
            apply_axis_theme(axis, theme_cfg)
    
    # ============ Plot 1: Reward vs Timesteps (all styles) ============
    ax = axes[0, 0]
    for style, phases in by_style.items():
        # Combine all phases (e.g., warmup, rlllm, ppo_only) and sort by timestep
        all_points = get_all_phase_points(phases)
        all_points.sort(key=lambda x: x.get('timestep_end', 0))
        
        timesteps = [p.get('timestep_end', 0) for p in all_points]
        rewards = [p.get('avg_env_reward', 0) for p in all_points]
        rewards = normalize_series(rewards, mode=normalize_mode)
        rewards = smooth_series(rewards, window=smooth_window)
        
        color = colors.get(style, None)
        ax.plot(
            timesteps,
            rewards,
            marker='o',
            linewidth=2.5,
            label=display_style_name(style),
            color=color,
            markersize=6,
        )
    
    ax.set_xlabel('Timesteps', fontsize=11, fontweight='bold')
    ylabel = 'Average Environment Reward'
    if normalize_mode == 'minmax':
        ylabel = 'Normalized Reward (min-max)'
    elif normalize_mode == 'zscore':
        ylabel = 'Normalized Reward (z-score)'
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title('Reward vs Timesteps (All Styles)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.35, linestyle='--', color=theme_cfg['grid'])
    ax.legend(loc='best', fontsize=10)
    
    # ============ Plot 2: Reward Model Loss vs Timesteps ============
    ax = axes[0, 1]
    for style, phases in by_style.items():
        rlllm_points = phases.get('rlhf', [])
        rlllm_points.sort(key=lambda x: x.get('timestep_end', 0))
        
        timesteps = [p.get('timestep_end', 0) for p in rlllm_points]
        rm_losses = [p.get('rm_final_loss') for p in rlllm_points]
        rm_losses = [l for l in rm_losses if l is not None]  # Filter None values
        
        if rm_losses:
            timesteps_valid = timesteps[-len(rm_losses):]
            color = colors.get(style, None)
            ax.plot(
                timesteps_valid,
                rm_losses,
                marker='s',
                linewidth=2.5,
                label=display_style_name(style),
                color=color,
                markersize=6,
            )
    
    ax.set_xlabel('Timesteps', fontsize=11, fontweight='bold')
    ax.set_ylabel('Reward Model Loss', fontsize=11, fontweight='bold')
    ax.set_title('Reward Model Loss Over Training (RLLLM only)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.35, linestyle='--', color=theme_cfg['grid'])
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize=10)
    
    # ============ Plot 3: Warmup vs RLLLM Comparison ============
    ax = axes[1, 0]
    bar_styles = [
        style for style, phases in by_style.items()
        if phases.get('warmup') or phases.get('rlhf')
    ]
    x_pos = np.arange(len(bar_styles))
    warmup_rewards = []
    rlllm_rewards = []
    warmup_bar_colors = []
    rlllm_bar_colors = []
    warmup_edge_colors = []
    rlllm_edge_colors = []
    
    for style in bar_styles:
        warmup_pts = by_style[style].get('warmup', [])
        rlllm_pts = by_style[style].get('rlhf', [])
        
        warmup_vals = [p.get('avg_env_reward', 0) for p in warmup_pts]
        rlllm_vals = [p.get('avg_env_reward', 0) for p in rlllm_pts]
        warmup_vals = normalize_series(warmup_vals, mode=normalize_mode)
        rlllm_vals = normalize_series(rlllm_vals, mode=normalize_mode)

        warmup_avg = np.mean(warmup_vals) if warmup_vals else 0
        rlllm_avg = np.mean(rlllm_vals) if rlllm_vals else 0
        
        warmup_rewards.append(warmup_avg)
        rlllm_rewards.append(rlllm_avg)

        # Keep ppo_only_policy red in bar charts; use blue for RLLLM phase otherwise.
        if style == 'ppo_only_policy':
            warmup_bar_colors.append(theme_cfg['true_fill'])
            rlllm_bar_colors.append(theme_cfg['true_fill'])
            warmup_edge_colors.append(theme_cfg['true_edge'])
            rlllm_edge_colors.append(theme_cfg['true_edge'])
        else:
            warmup_bar_colors.append(theme_cfg['warmup_fill'])
            rlllm_bar_colors.append(theme_cfg['rlllm_fill'])
            warmup_edge_colors.append(theme_cfg['warmup_edge'])
            rlllm_edge_colors.append(theme_cfg['rlllm_edge'])
    
    width = 0.35
    ax.bar(x_pos - width/2, warmup_rewards, width, label='Warmup Phase', color=warmup_bar_colors, edgecolor=warmup_edge_colors, linewidth=2)
    ax.bar(x_pos + width/2, rlllm_rewards, width, label='RLLLM Phase', color=rlllm_bar_colors, edgecolor=rlllm_edge_colors, linewidth=2)
    
    ax.set_xlabel('Prompt Style', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title('Warmup vs RLLLM Phase Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([display_style_name(s) for s in bar_styles], rotation=15, ha='right')
    ax.grid(True, alpha=0.35, axis='y', linestyle='--', color=theme_cfg['grid'])
    ax.legend(fontsize=10)
    
    # ============ Plot 4: Survival Metric (if available) or round progression ============
    ax = axes[1, 1]
    survival_key = 'episode_length_mean'
    has_per_round_survival = any(
        any(p.get(survival_key) is not None for p in get_all_phase_points(phases))
        for phases in by_style.values()
    )
    if has_per_round_survival:
        for style, phases in by_style.items():
            all_points = get_all_phase_points(phases)
            all_points.sort(key=lambda x: x.get('global_round', 0))

            rounds = []
            episode_means = []
            for p in all_points:
                val = p.get(survival_key)
                if val is None:
                    continue
                rounds.append(p.get('global_round', 0))
                episode_means.append(val)

            if not episode_means:
                continue

            episode_means = smooth_series(episode_means, window=smooth_window)

            color = colors.get(style, None)
            ax.plot(
                rounds,
                episode_means,
                marker='D',
                linewidth=2.5,
                label=display_style_name(style),
                color=color,
                markersize=6,
            )

        ax.set_xlabel('Round Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Episode Length Mean', fontsize=11, fontweight='bold')
        ax.set_title('Survival by Training Round', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.35, linestyle='--', color=theme_cfg['grid'])
        ax.legend(loc='best', fontsize=10)
    else:
        for style, phases in by_style.items():
            all_points = get_all_phase_points(phases)
            all_points.sort(key=lambda x: x.get('global_round', 0))

            rounds = [p.get('global_round', 0) for p in all_points]
            rewards = [p.get('avg_env_reward', 0) for p in all_points]
            rewards = normalize_series(rewards, mode=normalize_mode)
            rewards = smooth_series(rewards, window=smooth_window)

            color = colors.get(style, None)
            ax.plot(
                rounds,
                rewards,
                marker='D',
                linewidth=2.5,
                label=display_style_name(style),
                color=color,
                markersize=6,
            )

        ax.set_xlabel('Round Number', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title('Reward by Training Round', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.35, linestyle='--', color=theme_cfg['grid'])
        ax.legend(loc='best', fontsize=10)
    
    # ============ Build Title with Hyperparameters ============
    env_name = config.get('env_name', 'unknown').upper()
    warmup_rounds = config.get('warmup_rounds', 0)
    rlllm_rounds = config.get('rlhf_rounds', 0)
    train_rollout = config.get('training_rollout_steps', 0)
    eval_rollout = config.get('eval_rollout_steps', 0)
    seed = config.get('seed', 0)
    num_styles = len(config.get('styles', []))
    
    title_str = (
        f"{title_from_json}\n"
        f"Env: {env_name} | Warmup Rounds: {warmup_rounds} | RLLLM Rounds: {rlllm_rounds} | "
        f"Train Rollout: {train_rollout} | Eval Rollout: {eval_rollout}\n"
        f"Styles: {num_styles} | Seed: {seed} | Normalize: {normalize_mode} | Smooth: {smooth_window}"
    )
    
    fig.suptitle(title_str, fontsize=13, fontweight='bold', y=0.995, color=theme_cfg['text'])
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # ============ Save Figure ============
    os.makedirs(output_dir, exist_ok=True)
    
    # Create descriptive filename with hyperparameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_clean = env_name.lower().replace('/', '_')
    base_name = Path(json_file).stem
    
    param_str = (
        f"w{warmup_rounds}_r{rlllm_rounds}_"
        f"tr{train_rollout}_er{eval_rollout}_"
        f"s{num_styles}_seed{seed}"
    )
    
    output_stem = f"rlllm_{env_clean}_{base_name}_{param_str}_{timestamp}"
    output_filename = f"{output_dir}/{output_stem}.png"
    
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_filename}")

    if save_data:
        # Save the full source JSON with the same base name for consistent archival.
        os.makedirs(data_dir, exist_ok=True)
        data_filename = f"{data_dir}/{output_stem}.json"
        with open(data_filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Data saved to: {data_filename}")
    
    plt.close()
    
    # Also print a data summary
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    print(f"Title: {title_from_json}")
    print(f"Environment: {env_name}")
    print(f"Config:")
    print(f"  - Warmup Rounds: {warmup_rounds}")
    print(f"  - RLLLM Rounds: {rlllm_rounds}")
    print(f"  - Training Rollout Steps: {train_rollout}")
    print(f"  - Evaluation Rollout Steps: {eval_rollout}")
    print(f"  - Seed: {seed}")
    print(f"  - Prompt Styles: {num_styles}")
    print(f"\nPrompt Styles used:")
    for i, style in enumerate(config.get('styles', []), 1):
        print(f"  {i}. {display_style_name(style)}")
    print(f"\nTotal data points: {len(data.get('data_points', []))}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize RLLLM training results from JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_rlhf_json.py pendulem2.json
  python visualize_rlhf_json.py hopper_results.json --output_dir my_graphs
        """
    )
    
    parser.add_argument('json_file', type=str, help='Path to JSON file with training data')
    parser.add_argument('--output_dir', type=str, default='graphs',
                       help='Directory to save output plots (default: graphs)')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory to save archived JSON files (default: data)')
    parser.add_argument('--save-data', '--save_data', action='store_true',
                       help='If set, also save a same-named JSON copy into data_dir')
    parser.add_argument('--normalize', type=str, default='none', choices=['none', 'minmax', 'zscore'],
                       help='Optional reward normalization mode for plotted reward curves')
    parser.add_argument('--smooth-window', type=int, default=1,
                       help='Moving-average window for reward curves (>=1, default: 1)')
    parser.add_argument('--theme', type=str, default='light', choices=['light', 'dark'],
                       help='Plot theme (default: light)')
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.json_file):
        print(f"Error: File not found: {args.json_file}")
        exit(1)
    
    # Validate smoothing window
    if args.smooth_window < 1:
        print('Error: --smooth-window must be >= 1')
        exit(1)

    # Create visualization
    create_visualization(
        args.json_file,
        args.output_dir,
        args.data_dir,
        args.save_data,
        normalize_mode=args.normalize,
        smooth_window=args.smooth_window,
        theme=args.theme,
    )


if __name__ == '__main__':
    main()
