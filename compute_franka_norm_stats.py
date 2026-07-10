#!/usr/bin/env python
"""
Compute FRANAK Dataset Normalization Statistics

FRANAK data format:
- state (proprio): 8-dim [ee_pos(3), ee_ori(3), gripper_states(2)]
- actions: 7-dim [delta_xyz(3), delta_euler(3), gripper_cmd(1)]

Output format:
{
  "norm_stats": {
    "state": {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
    "actions": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}
  }
}

Usage:
    python compute_libero_norm_stats.py \\
        --data_dir /path/to/LIBERO/datasets \\
        --output ./norm_stats/libero_norm.json
"""

import argparse
import json
import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import h5py
from tqdm import tqdm


class RunningStats:
    """Compute running statistics for large datasets."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self._count = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._mean_of_squares = np.zeros(dim, dtype=np.float64)
        self._min = np.full(dim, np.inf, dtype=np.float64)
        self._max = np.full(dim, -np.inf, dtype=np.float64)
        
        # Sample collection for quantile computation
        self._samples: List[np.ndarray] = []
        self._max_samples = 10000000
        
    def update(self, batch: np.ndarray) -> None:
        """Update statistics."""
        batch = batch.reshape(-1, batch.shape[-1]).astype(np.float64)
        n = batch.shape[0]
        
        if n == 0:
            return
            
        # Update min/max
        batch_min = np.min(batch, axis=0)
        batch_max = np.max(batch, axis=0)
        self._min = np.minimum(self._min, batch_min)
        self._max = np.maximum(self._max, batch_max)
        
        # Collect samples for quantile computation
        # if len(self._samples) * 1000 < self._max_samples:
        #     sample_idx = np.random.choice(n, min(100, n), replace=False)
        #     self._samples.append(batch[sample_idx])

        self._samples.append(batch)
        
        # Update running mean and mean of squares
        batch_mean = np.mean(batch, axis=0)
        batch_mean_sq = np.mean(batch ** 2, axis=0)
        
        total = self._count + n
        self._mean = (self._mean * self._count + batch_mean * n) / total
        self._mean_of_squares = (self._mean_of_squares * self._count + batch_mean_sq * n) / total
        self._count = total
        
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """Get statistics."""
        if self._count < 2:
            raise ValueError("Need at least 2 samples to compute statistics")
            
        variance = self._mean_of_squares - self._mean ** 2
        std = np.sqrt(np.maximum(0, variance))
        
        # Compute quantiles
        all_samples = np.concatenate(self._samples, axis=0) if self._samples else np.zeros((1, self.dim))
        q01 = np.percentile(all_samples, 1, axis=0)
        q99 = np.percentile(all_samples, 99, axis=0)
        
        return {
            "mean": self._mean.astype(np.float32),
            "std": std.astype(np.float32),
            "q01": q01.astype(np.float32),
            "q99": q99.astype(np.float32),
            "min": self._min.astype(np.float32),
            "max": self._max.astype(np.float32),
            "count": int(self._count),
        }


def compute_norm_stats(
    data_dir: str,
    subsets: List[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute FRANAK dataset normalization statistics.
    
    Args:
        data_dir: FRANAK dataset root directory
        subsets: Subsets to include, default ["franka_hdf5", "franka_spatial"]
        output_path: Output JSON path
        
    Returns:
        Dictionary containing state and actions statistics
    """
    if subsets is None:
        subsets = ["franka-hdf5", "franka-spatial"]
    
    print(f"Computing FRANKA normalization statistics")
    print(f"   Data directory: {data_dir}")
    print(f"   Subsets: {subsets}")
    print(f"   State dimension: 12 [ee_pos(3), ee_ori(3), hand_joint(6)]")
    print(f"   Actions dimension: 12 [delta_xyz(3), delta_euler(3), hand_joint(6)]")
    
    # Initialize statistics
    state_stats = RunningStats(dim=12)
    action_stats = RunningStats(dim=12)
    
    total_demos = 0
    total_steps = 0
    
    # Iterate through all subsets
    for subset in subsets:
        subset_dir = os.path.join(data_dir, subset)
        if not os.path.exists(subset_dir):
            print(f"Warning: Skipping non-existent directory: {subset_dir}")
            continue
            
        h5_files = sorted(glob.glob(os.path.join(subset_dir, "*.hdf5")))
        print(f"\nProcessing {subset}: {len(h5_files)} files")
        
        for h5_path in tqdm(h5_files, desc=subset):
            try:
                with h5py.File(h5_path, "r") as f:
                    if "data" not in f:
                        continue
                    data_grp = f["data"]
                    
                    for demo_key in data_grp.keys():
                        demo = data_grp[demo_key]
                        
                        # Check required keys
                        if "arm_actions" not in demo["right"]:
                            continue

                        # Load data
                        arm_actions = np.array(demo["right"]["arm_actions"])  # [T, 6]
                        hand_actions = np.array(demo["right"]["hand_actions"])  # [T, 6]
                        
                        # Build state
                        ee_pos = np.array(demo["right"]["obs"]["ee_pos"]) 
                        ee_ori = np.array(demo["right"]["obs"]["ee_ori"]) 
                        hand_joint = np.array(demo["right"]["obs"]["hand_joint"]) 
                        
                        T = min(len(arm_actions), len(hand_actions), len(ee_pos), len(ee_ori), len(hand_joint))
                        
                        state = np.concatenate([
                            ee_pos[:T],
                            ee_ori[:T],
                            hand_joint[:T]
                        ], axis=-1).astype(np.float32)

                        actions = np.concatenate([
                            arm_actions[:T],
                            hand_actions[:T]
                        ], axis=-1).astype(np.float32)

                        # Update statistics
                        state_stats.update(state)
                        action_stats.update(actions)
                        
                        total_demos += 1
                        total_steps += T
                        
            except Exception as e:
                print(f"Error processing {h5_path}: {e}")
                continue
    
    print(f"\nStatistics computation complete")
    print(f"   Total demos: {total_demos}")
    print(f"   Total steps: {total_steps}")
    
    # Get statistics
    state_norm_stats = state_stats.get_statistics()
    action_norm_stats = action_stats.get_statistics()
    
    # Print results
    state_labels = ["ee_x", "ee_y", "ee_z", "ori_r", "ori_p", "ori_y", "hand_0", "hand_1", "hand_2", "hand_3", "hand_4", "hand_5"]
    action_labels = ["dx", "dy", "dz", "dr", "dp", "dyaw", "hand_0", "hand_1", "hand_2", "hand_3", "hand_4", "hand_5"]
    
    print(f"\nState (12-dim) statistics:")
    print(f"{'dim':<10} {'mean':>10} {'std':>10} {'q01':>10} {'q99':>10}")
    print("-" * 50)
    for i, label in enumerate(state_labels):
        print(f"{label:<10} {state_norm_stats['mean'][i]:>10.4f} {state_norm_stats['std'][i]:>10.4f} "
              f"{state_norm_stats['q01'][i]:>10.4f} {state_norm_stats['q99'][i]:>10.4f}")
    
    print(f"\nActions (12-dim) statistics:")
    print(f"{'dim':<10} {'mean':>10} {'std':>10} {'q01':>10} {'q99':>10}")
    print("-" * 50)
    for i, label in enumerate(action_labels):
        print(f"{label:<10} {action_norm_stats['mean'][i]:>10.4f} {action_norm_stats['std'][i]:>10.4f} "
              f"{action_norm_stats['q01'][i]:>10.4f} {action_norm_stats['q99'][i]:>10.4f}")
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "norm_stats": {
                "state": {
                    "mean": state_norm_stats["mean"].tolist(),
                    "std": state_norm_stats["std"].tolist(),
                    "q01": state_norm_stats["q01"].tolist(),
                    "q99": state_norm_stats["q99"].tolist(),
                },
                "actions": {
                    "mean": action_norm_stats["mean"].tolist(),
                    "std": action_norm_stats["std"].tolist(),
                    "q01": action_norm_stats["q01"].tolist(),
                    "q99": action_norm_stats["q99"].tolist(),
                },
            },
            "metadata": {
                "data_dir": data_dir,
                "subsets": subsets,
                "num_demos": total_demos,
                "num_steps": total_steps,
                "state_dim": 12,
                "action_dim": 12,
                "state_labels": state_labels,
                "action_labels": action_labels,
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
            
        print(f"\nSaved to: {output_path}")
        
    return {"state": state_norm_stats, "actions": action_norm_stats}


def main():
    parser = argparse.ArgumentParser(description="Compute FRANKA normalization statistics")
    parser.add_argument("--data_dir", type=str, required=False,
                        help="FRANKA dataset root directory")
    parser.add_argument("--subsets", type=str, nargs="+",
                        default=["franka-hdf5", "franka-spatial"],
                        help="Subsets to include (default 2 subsets)")
    parser.add_argument("--output", type=str,
                        default="./norm_stats/franka_norm.json",
                        help="Output file path")
    
    args = parser.parse_args()

    args.data_dir = "/home/keep/Desktop/project/X-RLinf/dataset"
    args.subsets = ["franka-l6-20260503"]
    args.output = "/home/keep/Desktop/project/X-RLinf/dataset/franka-l6-20260503/franka_norm.json"
    
    compute_norm_stats(
        data_dir=args.data_dir,
        subsets=args.subsets,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
