#!/usr/bin/env python3
"""
Visualize RLHF training results from JSON output files.
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
            by_style[style] = {'warmup': [], 'rlhf': []}
        by_style[style][point.get('phase', 'warmup')].append(point)
    
    return by_style


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
):
    """Create 4-panel visualization from JSON training data."""
    
    # Load data
    print(f"Loading {json_file}...")
    data = load_json_data(json_file)
    
    config = data.get('config', {})
    details = data.get('details', {})
    title_from_json = data.get('title', 'RLHF Training Results')
    
    # Parse data points by style
    by_style = parse_data_points(data)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors for different styles
    colors = {
        'balanced_goal_first': '#1f77b4',
        'checklist_style': '#ff7f0e',
        'minimal_stats_style': '#2ca02c',
    }
    
    # ============ Plot 1: Reward vs Timesteps (all styles) ============
    ax = axes[0, 0]
    for style, phases in by_style.items():
        # Combine warmup + rlhf data points and sort by timestep
        all_points = phases['warmup'] + phases['rlhf']
        all_points.sort(key=lambda x: x.get('timestep_end', 0))
        
        timesteps = [p.get('timestep_end', 0) for p in all_points]
        rewards = [p.get('avg_env_reward', 0) for p in all_points]
        rewards = normalize_series(rewards, mode=normalize_mode)
        rewards = smooth_series(rewards, window=smooth_window)
        
        color = colors.get(style, None)
        ax.plot(timesteps, rewards, marker='o', linewidth=2.5, label=style, color=color, markersize=6)
    
    ax.set_xlabel('Timesteps', fontsize=11, fontweight='bold')
    ylabel = 'Average Environment Reward'
    if normalize_mode == 'minmax':
        ylabel = 'Normalized Reward (min-max)'
    elif normalize_mode == 'zscore':
        ylabel = 'Normalized Reward (z-score)'
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title('Reward vs Timesteps (All Styles)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # ============ Plot 2: Reward Model Loss vs Timesteps ============
    ax = axes[0, 1]
    for style, phases in by_style.items():
        rlhf_points = phases['rlhf']
        rlhf_points.sort(key=lambda x: x.get('timestep_end', 0))
        
        timesteps = [p.get('timestep_end', 0) for p in rlhf_points]
        rm_losses = [p.get('rm_final_loss') for p in rlhf_points]
        rm_losses = [l for l in rm_losses if l is not None]  # Filter None values
        
        if rm_losses:
            timesteps_valid = timesteps[-len(rm_losses):]
            color = colors.get(style, None)
            ax.plot(timesteps_valid, rm_losses, marker='s', linewidth=2.5, label=style, 
                   color=color, markersize=6)
    
    ax.set_xlabel('Timesteps', fontsize=11, fontweight='bold')
    ax.set_ylabel('Reward Model Loss', fontsize=11, fontweight='bold')
    ax.set_title('Reward Model Loss Over Training (RLHF only)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # ============ Plot 3: Warmup vs RLHF Comparison ============
    ax = axes[1, 0]
    x_pos = np.arange(len(by_style))
    warmup_rewards = []
    rlhf_rewards = []
    
    for style in by_style.keys():
        warmup_pts = by_style[style]['warmup']
        rlhf_pts = by_style[style]['rlhf']
        
        warmup_vals = [p.get('avg_env_reward', 0) for p in warmup_pts]
        rlhf_vals = [p.get('avg_env_reward', 0) for p in rlhf_pts]
        warmup_vals = normalize_series(warmup_vals, mode=normalize_mode)
        rlhf_vals = normalize_series(rlhf_vals, mode=normalize_mode)

        warmup_avg = np.mean(warmup_vals) if warmup_vals else 0
        rlhf_avg = np.mean(rlhf_vals) if rlhf_vals else 0
        
        warmup_rewards.append(warmup_avg)
        rlhf_rewards.append(rlhf_avg)
    
    width = 0.35
    ax.bar(x_pos - width/2, warmup_rewards, width, label='Warmup Phase', color='#ccccff', edgecolor='blue', linewidth=2)
    ax.bar(x_pos + width/2, rlhf_rewards, width, label='RLHF Phase', color='#ffcccc', edgecolor='red', linewidth=2)
    
    ax.set_xlabel('Prompt Style', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title('Warmup vs RLHF Phase Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(by_style.keys()), rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(fontsize=10)
    
    # ============ Plot 4: Training Progression (round by round) ============
    ax = axes[1, 1]
    for style, phases in by_style.items():
        all_points = phases['warmup'] + phases['rlhf']
        all_points.sort(key=lambda x: x.get('global_round', 0))
        
        rounds = [p.get('global_round', 0) for p in all_points]
        rewards = [p.get('avg_env_reward', 0) for p in all_points]
        rewards = normalize_series(rewards, mode=normalize_mode)
        rewards = smooth_series(rewards, window=smooth_window)
        
        color = colors.get(style, None)
        ax.plot(rounds, rewards, marker='D', linewidth=2.5, label=style, color=color, markersize=6)
    
    ax.set_xlabel('Round Number', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title('Reward by Training Round', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # ============ Build Title with Hyperparameters ============
    env_name = config.get('env_name', 'unknown').upper()
    warmup_rounds = config.get('warmup_rounds', 0)
    rlhf_rounds = config.get('rlhf_rounds', 0)
    train_rollout = config.get('training_rollout_steps', 0)
    eval_rollout = config.get('eval_rollout_steps', 0)
    seed = config.get('seed', 0)
    num_styles = len(config.get('styles', []))
    
    title_str = (
        f"{title_from_json}\n"
        f"Env: {env_name} | Warmup Rounds: {warmup_rounds} | RLHF Rounds: {rlhf_rounds} | "
        f"Train Rollout: {train_rollout} | Eval Rollout: {eval_rollout}\n"
        f"Styles: {num_styles} | Seed: {seed} | Normalize: {normalize_mode} | Smooth: {smooth_window}"
    )
    
    fig.suptitle(title_str, fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # ============ Save Figure ============
    os.makedirs(output_dir, exist_ok=True)
    
    # Create descriptive filename with hyperparameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_clean = env_name.lower().replace('/', '_')
    base_name = Path(json_file).stem
    
    param_str = (
        f"w{warmup_rounds}_r{rlhf_rounds}_"
        f"tr{train_rollout}_er{eval_rollout}_"
        f"s{num_styles}_seed{seed}"
    )
    
    output_stem = f"rlhf_{env_clean}_{base_name}_{param_str}_{timestamp}"
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
    print(f"  - RLHF Rounds: {rlhf_rounds}")
    print(f"  - Training Rollout Steps: {train_rollout}")
    print(f"  - Evaluation Rollout Steps: {eval_rollout}")
    print(f"  - Seed: {seed}")
    print(f"  - Prompt Styles: {num_styles}")
    print(f"\nPrompt Styles used:")
    for i, style in enumerate(config.get('styles', []), 1):
        print(f"  {i}. {style}")
    print(f"\nTotal data points: {len(data.get('data_points', []))}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize RLHF training results from JSON files',
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
    )


if __name__ == '__main__':
    main()
