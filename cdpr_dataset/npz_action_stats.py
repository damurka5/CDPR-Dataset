#!/usr/bin/env python3
"""npz_action_stats.py

Compute dataset-wide and per-episode action statistics directly from trajectory_data.npz files.

This is useful to debug cases where actions in TFRecords are mostly zeros, by recomputing
actions from different absolute sources (e.g., target_position vs ee_position).

It mirrors export_to_rlds_fixed2.py's action construction:
  abs5 = [x,y,z,yaw,grip]
  act[t] = abs5[t+1] - abs5[t]  (last step action forced to 0)
  yaw delta wrapped to [-pi, pi]

Outputs:
  - global_action_stats.json
  - episode_stats.csv
  - worst_episodes_by_zero_frac.txt
  - plots/*.png

Example:
  python npz_action_stats.py --episode_root /path/to/episodes --xyz_source ee_position --out out_ee
  python npz_action_stats.py --episode_root /path/to/episodes --xyz_source target_position --out out_target
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _find_xyz_key(npz_keys: List[str], requested: str) -> str:
    """Return the key name to use for xyz."""
    if requested in npz_keys:
        return requested

    # common aliases
    aliases = {
        "ee_position": [
            "ee_position", "ee_pos", "ee_xyz", "ee_position_world", "ee_position_abs",
            "end_effector_position", "eef_position", "eef_pos", "ee_translation"
        ],
        "target_position": [
            "target_position", "target_pos", "target_xyz", "goal_position", "setpoint_position"
        ],
    }

    for cand in aliases.get(requested, []):
        if cand in npz_keys:
            return cand

    # fuzzy contains match
    req_tokens = requested.split("_")
    scored: List[Tuple[int, str]] = []
    for k in npz_keys:
        score = sum(1 for t in req_tokens if t in k)
        if score > 0:
            scored.append((score, k))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]

    raise KeyError(
        f"Could not find xyz source '{requested}' in npz. Available keys: {npz_keys}"
    )


def _wrap_to_pi(dyaw: np.ndarray) -> np.ndarray:
    return (dyaw + np.pi) % (2 * np.pi) - np.pi


def _streaming_update(stream: Dict[str, np.ndarray], x: np.ndarray):
    """Update streaming stats for a batch of vectors x with shape (N, D)."""
    if x.size == 0:
        return
    if stream["count"] == 0:
        stream["min"] = np.min(x, axis=0)
        stream["max"] = np.max(x, axis=0)
    else:
        stream["min"] = np.minimum(stream["min"], np.min(x, axis=0))
        stream["max"] = np.maximum(stream["max"], np.max(x, axis=0))

    stream["sum"] += np.sum(x, axis=0)
    stream["sumsq"] += np.sum(x * x, axis=0)
    stream["count"] += x.shape[0]

def _streaming_finalize(stream: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
    n = int(stream["count"])
    if n == 0:
        return {}
    mean = stream["sum"] / n
    var = stream["sumsq"] / n - mean * mean
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    return {
        "count": n,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": stream["min"].tolist(),
        "max": stream["max"].tolist(),
    }


def _list_npz_files(episode_root: Path, recursive: bool) -> List[Path]:
    if episode_root.is_file() and episode_root.suffix == ".npz":
        return [episode_root]

    pattern = "**/trajectory_data.npz" if recursive else "*/trajectory_data.npz"
    return sorted(episode_root.glob(pattern))


def compute_actions_from_npz(
    npz_path: Path,
    xyz_source: str,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Return act array (N,5) and metadata."""

    logs = np.load(npz_path, allow_pickle=True)
    keys = list(logs.files)

    if "control_signals" not in logs:
        raise KeyError(f"Missing 'control_signals' in {npz_path}. keys={keys}")
    ctrl_all = np.asarray(logs["control_signals"], dtype=np.float32)
    if ctrl_all.ndim != 2 or ctrl_all.shape[1] < 2:
        raise ValueError(f"control_signals has unexpected shape {ctrl_all.shape} in {npz_path}")

    xyz_key = _find_xyz_key(keys, xyz_source)
    xyz_all = np.asarray(logs[xyz_key], dtype=np.float32)
    if xyz_all.ndim != 2 or xyz_all.shape[1] < 3:
        raise ValueError(f"{xyz_key} has unexpected shape {xyz_all.shape} in {npz_path}")

    # Optional timestamp for alignment
    if "timestamp" in logs:
        ts_all = np.asarray(logs["timestamp"]).reshape(-1)
        N = int(min(len(ts_all), len(xyz_all), len(ctrl_all)))
    else:
        N = int(min(len(xyz_all), len(ctrl_all)))

    if N < 2:
        raise ValueError(f"Episode too short after alignment N={N} in {npz_path}")

    xyz = xyz_all[:N, :3]
    ctrl = ctrl_all[:N]

    yaw = ctrl[:, -2]
    gr = ctrl[:, -1]

    abs5 = np.zeros((N, 5), np.float32)
    abs5[:, 0:3] = xyz
    abs5[:, 3] = yaw
    abs5[:, 4] = gr

    act = np.zeros_like(abs5)
    act[:-1] = abs5[1:] - abs5[:-1]
    act[:-1, 3] = _wrap_to_pi(act[:-1, 3])
    act[-1] = 0.0

    meta = {
        "xyz_key": xyz_key,
        "has_timestamp": str("timestamp" in logs),
        "ctrl_shape": str(ctrl_all.shape),
        "xyz_shape": str(xyz_all.shape),
    }
    return act, meta
# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_root", type=str, required=True,
                    help="Directory containing episode folders (each with trajectory_data.npz), or a single .npz file")
    ap.add_argument("--recursive", action="store_true", help="Search recursively for trajectory_data.npz")
    ap.add_argument("--xyz_source", type=str, default="target_position",
                    help="Which npz key to use for xyz. Common: target_position or ee_position")
    ap.add_argument("--out", type=str, default="npz_stats_out", help="Output directory")
    ap.add_argument("--eps", type=float, default=1e-6, help="Per-component near-zero threshold")
    ap.add_argument("--eps_norm", type=float, default=1e-6, help="Vector-norm near-zero threshold")
    ap.add_argument("--max_episodes", type=int, default=0, help="If >0, limit episodes processed")
    ap.add_argument("--topk", type=int, default=50, help="How many worst episodes to list")

    args = ap.parse_args()

    episode_root = Path(args.episode_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    npz_files = _list_npz_files(episode_root, args.recursive)
    if not npz_files:
        raise SystemExit(f"No trajectory_data.npz found under {episode_root} (recursive={args.recursive})")
    if args.max_episodes and args.max_episodes > 0:
        npz_files = npz_files[: args.max_episodes]

    D = 5
    dof_names = ["x", "y", "z", "w", "g"]

    # Streaming stats for action components
    stream = {
        "count": 0,
        "sum": np.zeros((D,), np.float64),
        "sumsq": np.zeros((D,), np.float64),
        "min": np.full((D,), np.inf, np.float64),
        "max": np.full((D,), -np.inf, np.float64),
    }

    # Global counters
    total_steps = 0
    exact0_norm = 0
    near0_norm = 0

    exact0_dof = np.zeros((D,), np.int64)
    near0_dof = np.zeros((D,), np.int64)

    # Collect samples for plotting (cap to avoid huge memory)
    sample_cap = 2_000_000
    samples = []  # list of (n,5)
    sample_count = 0

    episode_rows = []
    for p in tqdm(npz_files, desc=f"Scanning npz (xyz_source={args.xyz_source})"):
        ep_name = p.parent.name
        try:
            act, meta = compute_actions_from_npz(p, args.xyz_source)
        except Exception as e:
            episode_rows.append({
                "episode": ep_name,
                "npz": str(p),
                "status": "error",
                "error": str(e),
            })
            continue

        N = act.shape[0]
        total_steps += N

        # streaming stats
        _streaming_update(stream, act.astype(np.float64))

        # norms
        norm = np.linalg.norm(act, axis=1)
        exact0_mask = norm == 0.0
        near0_mask = norm < args.eps_norm

        exact0_norm += int(np.sum(exact0_mask))
        near0_norm += int(np.sum(near0_mask))

        # dof exact/near
        exact0_mask_d = (act == 0.0)
        near0_mask_d = (np.abs(act) < args.eps)

        exact0_dof += np.sum(exact0_mask_d, axis=0).astype(np.int64)
        near0_dof += np.sum(near0_mask_d, axis=0).astype(np.int64)

        # store sample
        if sample_count < sample_cap:
            take = min(sample_cap - sample_count, N)
            if take > 0:
                samples.append(act[:take].copy())
                sample_count += take

        # per-episode summary
        row = {
            "episode": ep_name,
            "npz": str(p),
            "status": "ok",
            "steps": N,
            "zero_norm_frac": float(np.mean(exact0_mask)),
            "near0_norm_frac": float(np.mean(near0_mask)),
            "mean_norm": float(np.mean(norm)),
            "p50_norm": float(np.percentile(norm, 50)),
            "p95_norm": float(np.percentile(norm, 95)),
            "max_norm": float(np.max(norm)),
            "xyz_key": meta.get("xyz_key", ""),
            "ctrl_shape": meta.get("ctrl_shape", ""),
            "xyz_shape": meta.get("xyz_shape", ""),
        }
        for i, n in enumerate(dof_names):
            row[f"{n}_exact0_frac"] = float(np.mean(act[:, i] == 0.0))
            row[f"{n}_near0_frac"] = float(np.mean(np.abs(act[:, i]) < args.eps))
            row[f"{n}_min"] = float(np.min(act[:, i]))
            row[f"{n}_max"] = float(np.max(act[:, i]))
            row[f"{n}_std"] = float(np.std(act[:, i]))
        episode_rows.append(row)

    # Global summary
    if total_steps == 0:
        raise SystemExit("No valid episodes processed.")
    global_stats = _streaming_finalize(stream)
    global_stats.update({
        "steps": int(total_steps),
        "xyz_source_requested": args.xyz_source,
        "eps": float(args.eps),
        "eps_norm": float(args.eps_norm),
        "exact_zero_fraction_norm": float(exact0_norm / total_steps),
        "near_zero_fraction_norm": float(near0_norm / total_steps),
        "exact_zero_fraction_dof": (exact0_dof / total_steps).astype(float).tolist(),
        "near_zero_fraction_dof": (near0_dof / total_steps).astype(float).tolist(),
        "dof_names": dof_names,
        "episodes_total": int(len(npz_files)),
        "episodes_ok": int(sum(1 for r in episode_rows if r.get("status") == "ok")),
        "episodes_error": int(sum(1 for r in episode_rows if r.get("status") == "error")),
    })

    (out_dir / "global_action_stats.json").write_text(json.dumps(global_stats, indent=2))

    df = pd.DataFrame(episode_rows)
    df.to_csv(out_dir / "episode_stats.csv", index=False)

    # worst episodes list
    ok_df = df[df["status"] == "ok"].copy()
    ok_df.sort_values(["zero_norm_frac", "near0_norm_frac", "steps"], ascending=[False, False, False], inplace=True)
    worst = ok_df.head(args.topk)
    lines = []
    for _, r in worst.iterrows():
        lines.append(f"{r['zero_norm_frac']:.4f}\tsteps={int(r['steps'])}\t{r['episode']}\t{r['npz']}")
    (out_dir / "worst_episodes_by_zero_frac.txt").write_text("\n".join(lines) + "\n")

    # ---- plots ----
    if samples:
        A = np.concatenate(samples, axis=0)
    else:
        A = np.zeros((0, D), np.float32)

    if A.shape[0] > 0:
        # hist per dof
        for i, name in enumerate(dof_names):
            plt.figure()
            plt.hist(A[:, i], bins=200)
            plt.title(f"Action histogram: {name} (xyz_source={global_stats['xyz_source_requested']})")
            plt.xlabel("value")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(plots_dir / f"hist_{name}.png", dpi=160)
            plt.close()

        # norm hist
        nrm = np.linalg.norm(A, axis=1)
        plt.figure()
        plt.hist(nrm, bins=200)
        plt.title(f"Action norm histogram (xyz_source={global_stats['xyz_source_requested']})")
        plt.xlabel("||a||_2")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(plots_dir / "hist_norm.png", dpi=160)
        plt.close()

        # logabs hist per dof (helps see tiny values)
        for i, name in enumerate(dof_names):
            v = np.abs(A[:, i])
            v = np.maximum(v, 1e-12)
            lv = np.log10(v)
            plt.figure()
            plt.hist(lv, bins=200)
            plt.title(f"log10(|action|) histogram: {name}")
            plt.xlabel("log10(|a|)")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(plots_dir / f"logabs_{name}.png", dpi=160)
            plt.close()

        # correlation heatmap (components)
        C = np.corrcoef(A.T)
        plt.figure()
        plt.imshow(C, aspect="auto")
        plt.title("Action component correlation")
        plt.xticks(range(D), dof_names)
        plt.yticks(range(D), dof_names)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(plots_dir / "corr_components.png", dpi=160)
        plt.close()

    # global printout
    print("\n--- GLOBAL SUMMARY (NPZ-derived) ---")
    print(f"episodes ok/error: {global_stats['episodes_ok']}/{global_stats['episodes_error']}")
    print(f"steps: {global_stats['steps']}")
    print(f"near-zero (norm < eps_norm): {global_stats['near_zero_fraction_norm']:.4f}")
    print(f"exact-zero (norm == 0):      {global_stats['exact_zero_fraction_norm']:.4f}")
    for i, name in enumerate(dof_names):
        print(
            f"{name:>8s} | near0(abs<eps)={global_stats['near_zero_fraction_dof'][i]:.4f} "
            f"| exact0={global_stats['exact_zero_fraction_dof'][i]:.4f}"
        )


if __name__ == "__main__":
    main()
