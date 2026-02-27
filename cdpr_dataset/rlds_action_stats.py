#!/usr/bin/env python3
"""
rlds_action_stats.py

Dataset diagnostics for RLDS-style TFRecords written step-by-step as tf.train.Example
(as in export_to_rlds_fixed2.py).

What it does:
- Reads one or many .tfrecord files (each file typically = one demonstration / episode).
- Extracts the 'action' feature (float vector, default dim=5).
- Computes:
  * Global streaming stats across all steps (mean/std/min/max, zero fractions).
  * Per-episode stats (zero fraction, mean abs, norm distribution, etc).
- Saves plots (histograms per DoF, L2 norm histogram, log-abs hist, correlation heatmap),
  plus CSV/JSON summaries, and text files listing the "worst" demonstrations.

Typical use:
  python rlds_action_stats.py --tfrecord_dir /path/to/tfrecords --out stats_out

Inspect specific files:
  python rlds_action_stats.py --files demo1.tfrecord demo2.tfrecord --out stats_two

Notes:
- By default, "near-zero" means abs(action_i) < eps per DoF and ||action||_2 < eps_norm overall.
- Set eps/eps_norm based on your action scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# Matplotlib is used only for saving figures (no seaborn).
import matplotlib.pyplot as plt


# ------------------------- Utilities -------------------------

def list_tfrecords_from_dir(tfrecord_dir: Path, recursive: bool = True) -> List[Path]:
    exts = {".tfrecord", ".tfrecords", ".tfrec", ".tfr"}
    if recursive:
        files = [p for p in tfrecord_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in tfrecord_dir.glob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)

def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# ------------------------- Streaming stats -------------------------

@dataclass
class StreamingMoments:
    """Welford moments for vector data."""
    dim: int
    n: int = 0
    mean: Optional[np.ndarray] = None
    M2: Optional[np.ndarray] = None
    minv: Optional[np.ndarray] = None
    maxv: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.mean is None:
            self.mean = np.zeros((self.dim,), dtype=np.float64)
        if self.M2 is None:
            self.M2 = np.zeros((self.dim,), dtype=np.float64)
        if self.minv is None:
            self.minv = np.full((self.dim,), np.inf, dtype=np.float64)
        if self.maxv is None:
            self.maxv = np.full((self.dim,), -np.inf, dtype=np.float64)

    def update_batch(self, x: np.ndarray) -> None:
        """Update with batch x shape [B, dim]."""
        if x.size == 0:
            return
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected batch shape [B,{self.dim}], got {x.shape}")

        b = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_M2 = ((x - batch_mean) ** 2).sum(axis=0)

        if self.n == 0:
            self.mean = batch_mean
            self.M2 = batch_M2
            self.n = b
        else:
            n0 = self.n
            n1 = n0 + b
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * (b / n1)
            self.M2 = self.M2 + batch_M2 + (delta ** 2) * (n0 * b / n1)
            self.n = n1

        self.minv = np.minimum(self.minv, x.min(axis=0))
        self.maxv = np.maximum(self.maxv, x.max(axis=0))

    def finalize(self) -> dict:
        if self.n <= 1:
            var = np.full((self.dim,), np.nan, dtype=np.float64)
        else:
            var = self.M2 / (self.n - 1)
        std = np.sqrt(var)
        return {
            "count": int(self.n),
            "mean": self.mean.astype(np.float64).tolist(),
            "std": std.astype(np.float64).tolist(),
            "min": self.minv.astype(np.float64).tolist(),
            "max": self.maxv.astype(np.float64).tolist(),
        }


class Reservoir:
    """Reservoir sampling for rows of fixed-dim vectors."""
    def __init__(self, k: int, dim: int, seed: int = 0):
        self.k = int(k)
        self.dim = int(dim)
        self.n_seen = 0
        self.buf = np.empty((0, dim), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def add_batch(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Reservoir expected [B,{self.dim}], got {x.shape}")

        b = x.shape[0]
        # Fill buffer if not full
        if self.buf.shape[0] < self.k:
            take = min(self.k - self.buf.shape[0], b)
            self.buf = np.concatenate([self.buf, x[:take]], axis=0)
            self.n_seen += take
            x = x[take:]
            b = x.shape[0]
            if b == 0:
                return

        # Standard reservoir replacement for remaining rows (vectorized per-batch)
        # For each incoming item at global index i, draw r ~ Uniform{0..i}. If r < k, replace buf[r].
        start = self.n_seen
        idx = np.arange(start, start + b, dtype=np.int64)
        # r_i = floor(u_i * (i+1))
        r = np.floor(self.rng.random(b) * (idx + 1)).astype(np.int64)
        mask = r < self.k
        if np.any(mask):
            self.buf[r[mask]] = x[mask]
        self.n_seen += b

    def get(self) -> np.ndarray:
        return self.buf.copy()


# ------------------------- TFRecord reading -------------------------

def iter_action_batches_fixedlen(tfrecord_path: Path, action_dim: int, batch_size: int) -> Iterable[np.ndarray]:
    feature_desc = {
        "action": tf.io.FixedLenFeature([action_dim], tf.float32, default_value=[0.0] * action_dim),
    }
    ds = tf.data.TFRecordDataset(str(tfrecord_path))
    ds = ds.batch(batch_size)
    for raw_batch in ds:
        parsed = tf.io.parse_example(raw_batch, feature_desc)
        yield parsed["action"].numpy()

def read_actions_from_tfrecord(tfrecord_path: Path, action_dim: int, batch_size: int) -> np.ndarray:
    """
    Returns actions as np.ndarray [N, action_dim].

    Tries fast FixedLen parsing first. If that fails, falls back to tf.train.Example parsing.
    """
    try:
        chunks = []
        for a in iter_action_batches_fixedlen(tfrecord_path, action_dim, batch_size):
            chunks.append(a.astype(np.float32))
        if not chunks:
            return np.zeros((0, action_dim), dtype=np.float32)
        return np.concatenate(chunks, axis=0)
    except Exception:
        # Fallback: robust but slower
        out = []
        for raw in tf.data.TFRecordDataset(str(tfrecord_path)):
            ex = tf.train.Example.FromString(raw.numpy())
            fl = ex.features.feature.get("action")
            if fl is None:
                continue
            vals = list(fl.float_list.value)
            if len(vals) == 0:
                vals = [0.0] * action_dim
            if len(vals) != action_dim:
                # Pad/trim to be safe; also helps reveal schema issues in stats later.
                vals = (vals + [0.0] * action_dim)[:action_dim]
            out.append(vals)
        if len(out) == 0:
            return np.zeros((0, action_dim), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)


# ------------------------- Episode stats -------------------------

@dataclass
class EpisodeStats:
    file: str
    steps: int
    # overall (L2)
    zero_frac_norm: float
    exact_zero_frac_norm: float
    mean_norm: float
    median_norm: float
    p95_norm: float
    mean_abs_norm: float
    # per dof
    zero_frac_dof: List[float]
    exact_zero_frac_dof: List[float]
    mean_abs_dof: List[float]
    std_dof: List[float]
    min_dof: List[float]
    max_dof: List[float]


def compute_episode_stats(actions: np.ndarray, eps: float, eps_norm: float, file: str) -> EpisodeStats:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"actions must be [N,dim], got {a.shape}")
    N, D = a.shape
    if N == 0:
        return EpisodeStats(
            file=file, steps=0,
            zero_frac_norm=float("nan"),
            exact_zero_frac_norm=float("nan"),
            mean_norm=float("nan"),
            median_norm=float("nan"),
            p95_norm=float("nan"),
            mean_abs_norm=float("nan"),
            zero_frac_dof=[float("nan")] * D,
            exact_zero_frac_dof=[float("nan")] * D,
            mean_abs_dof=[float("nan")] * D,
            std_dof=[float("nan")] * D,
            min_dof=[float("nan")] * D,
            max_dof=[float("nan")] * D,
        )

    abs_a = np.abs(a)
    norm = np.linalg.norm(a, axis=1)

    zero_frac_dof = (abs_a < eps).mean(axis=0)
    exact_zero_frac_dof = (a == 0.0).mean(axis=0)

    zero_frac_norm = (norm < eps_norm).mean()
    exact_zero_frac_norm = (norm == 0.0).mean()

    return EpisodeStats(
        file=file,
        steps=int(N),
        zero_frac_norm=float(zero_frac_norm),
        exact_zero_frac_norm=float(exact_zero_frac_norm),
        mean_norm=float(norm.mean()),
        median_norm=float(np.median(norm)),
        p95_norm=float(np.quantile(norm, 0.95)),
        mean_abs_norm=float(abs_a.mean(axis=1).mean()),
        zero_frac_dof=[float(x) for x in zero_frac_dof.tolist()],
        exact_zero_frac_dof=[float(x) for x in exact_zero_frac_dof.tolist()],
        mean_abs_dof=[float(x) for x in abs_a.mean(axis=0).tolist()],
        std_dof=[float(x) for x in a.std(axis=0, ddof=1).tolist()] if N > 1 else [0.0] * D,
        min_dof=[float(x) for x in a.min(axis=0).tolist()],
        max_dof=[float(x) for x in a.max(axis=0).tolist()],
    )


# ------------------------- Plotting -------------------------

def save_histogram(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 200,
                   range_: Optional[Tuple[float, float]] = None, logy: bool = False) -> None:
    plt.figure()
    plt.hist(values, bins=bins, range=range_)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_logabs_hist(values: np.ndarray, title: str, out_path: Path, bins: int = 200) -> None:
    # log10(|x| + tiny)
    tiny = 1e-12
    v = np.log10(np.abs(values) + tiny)
    plt.figure()
    plt.hist(v, bins=bins)
    plt.title(title)
    plt.xlabel("log10(|action| + 1e-12)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_corr_heatmap(sample_actions: np.ndarray, dof_names: List[str], out_path: Path) -> None:
    a = np.asarray(sample_actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2:
        return
    C = np.corrcoef(a.T)
    plt.figure()
    plt.imshow(C, interpolation="nearest")
    plt.title("Action DoF correlation (sample)")
    plt.xticks(range(len(dof_names)), dof_names, rotation=45, ha="right")
    plt.yticks(range(len(dof_names)), dof_names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_episode_bar(values: List[float], labels: List[str], title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(max(8, 0.35 * len(values)), 4))
    x = np.arange(len(values))
    plt.bar(x, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, labels, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_timeseries(actions: np.ndarray, dof_names: List[str], out_path: Path, title: str) -> None:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2:
        return
    T, D = a.shape
    plt.figure(figsize=(10, 5))
    for i in range(D):
        plt.plot(a[:, i], label=dof_names[i])
    plt.title(title)
    plt.xlabel("t (step index)")
    plt.ylabel("action")
    plt.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_norm_timeseries(actions: np.ndarray, out_path: Path, title: str) -> None:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2:
        return
    norm = np.linalg.norm(a, axis=1)
    plt.figure(figsize=(10, 4))
    plt.plot(norm)
    plt.title(title)
    plt.xlabel("t (step index)")
    plt.ylabel("||action||_2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord_dir", type=str, default="/root/repo/cdpr_synth_10hz/libero_spatial_no_noops/tfrecords_human_control_fixed",
                    help="Directory containing .tfrecord files (each file typically one episode).")
    ap.add_argument("--files", type=str, nargs="*", default=None,
                    help="Explicit list of tfrecord files to analyze (overrides --tfrecord_dir).")
    ap.add_argument("--recursive", action="store_true", help="Recursively search --tfrecord_dir.")
    ap.add_argument("--action_dim", type=int, default=5, help="Action vector dimension (default: 5).")
    ap.add_argument("--dof_names", type=str, nargs="*", default="xyzwg",
                    help="Names for DoFs (e.g. x y z yaw grip). Defaults to dof0..dofN.")
    ap.add_argument("--eps", type=float, default=1e-4,
                    help="Near-zero threshold per DoF: abs(action_i) < eps. (default: 1e-4)")
    ap.add_argument("--eps_norm", type=float, default=None,
                    help="Near-zero threshold for overall L2 norm: ||a||_2 < eps_norm. Default: sqrt(D)*eps")
    ap.add_argument("--batch_size", type=int, default=8192, help="TFRecord parse batch size.")
    ap.add_argument("--sample_size", type=int, default=300000,
                    help="Reservoir sample size for plots/quantiles (default: 300k).")
    ap.add_argument("--top_k", type=int, default=30, help="How many 'worst' episodes to list/plot.")
    ap.add_argument("--inspect", type=str, nargs="*", default=None,
                    help="Specific tfrecord files to additionally plot as timeseries.")
    ap.add_argument("--out", type=str, default="action_stats_out", help="Output directory.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_out_dir(out_dir)

    D = int(args.action_dim)
    dof_names = args.dof_names if args.dof_names else [f"dof{i}" for i in range(D)]
    if len(dof_names) != D:
        raise SystemExit(f"--dof_names must have length {D}, got {len(dof_names)}")

    eps = float(args.eps)
    eps_norm = float(args.eps_norm) if args.eps_norm is not None else float(math.sqrt(D) * eps)

    # Collect tfrecords
    if args.files and len(args.files) > 0:
        files = [Path(f) for f in args.files]
    else:
        if args.tfrecord_dir is None:
            raise SystemExit("Provide either --tfrecord_dir or --files ...")
        tfdir = Path(args.tfrecord_dir)
        if not tfdir.exists():
            raise SystemExit(f"tfrecord_dir does not exist: {tfdir}")
        files = list_tfrecords_from_dir(tfdir, recursive=args.recursive)

    files = [p for p in files if p.exists()]
    if len(files) == 0:
        raise SystemExit("No tfrecord files found.")

    print(f"Found {len(files)} tfrecord file(s).")
    print(f"Action dim={D} | eps={eps:g} | eps_norm={eps_norm:g}")
    print(f"Output -> {out_dir.resolve()}")

    # Global streaming stats
    moments = StreamingMoments(dim=D)
    # also compute streaming stats for norms (scalar)
    norm_moments = StreamingMoments(dim=1)

    # Near-zero counters
    total_steps = 0
    near0_dof = np.zeros((D,), dtype=np.int64)
    exact0_dof = np.zeros((D,), dtype=np.int64)
    near0_norm = 0
    exact0_norm = 0

    # Reservoir sample for plots/quantiles/correlation
    reservoir = Reservoir(k=int(args.sample_size), dim=D, seed=args.seed)

    episode_stats: List[EpisodeStats] = []

    for i, f in enumerate(files):
        actions = read_actions_from_tfrecord(f, action_dim=D, batch_size=args.batch_size)
        if actions.shape[0] == 0:
            print(f"[warn] empty: {f}")
            continue

        # Per-episode
        es = compute_episode_stats(actions, eps=eps, eps_norm=eps_norm, file=str(f))
        episode_stats.append(es)

        # Global updates
        moments.update_batch(actions)
        reservoir.add_batch(actions)

        # Norm batch
        norms = np.linalg.norm(actions.astype(np.float64), axis=1, keepdims=True)
        norm_moments.update_batch(norms)

        abs_a = np.abs(actions)
        near0_dof += (abs_a < eps).sum(axis=0).astype(np.int64)
        exact0_dof += (actions == 0.0).sum(axis=0).astype(np.int64)
        near0_norm += int((norms[:, 0] < eps_norm).sum())
        exact0_norm += int((norms[:, 0] == 0.0).sum())

        total_steps += int(actions.shape[0])

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(files)} files | steps={total_steps}")

    if total_steps == 0:
        raise SystemExit("No steps parsed from tfrecords.")

    # Global summary
    global_stats = {
        "action_dim": D,
        "dof_names": dof_names,
        "eps": eps,
        "eps_norm": eps_norm,
        "num_files": len(files),
        "num_steps": int(total_steps),
        "streaming_moments": moments.finalize(),
        "streaming_norm_moments": norm_moments.finalize(),
        "near_zero_fraction_dof": (near0_dof / total_steps).astype(float).tolist(),
        "exact_zero_fraction_dof": (exact0_dof / total_steps).astype(float).tolist(),
        "near_zero_fraction_norm": float(near0_norm / total_steps),
        "exact_zero_fraction_norm": float(exact0_norm / total_steps),
    }

    # Sample-derived quantiles + suggested histogram range
    sample = reservoir.get()
    if sample.shape[0] >= 10:
        q = {}
        for qi in [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]:
            q[str(qi)] = np.quantile(sample, qi, axis=0).astype(np.float64).tolist()
        global_stats["sample_quantiles_per_dof"] = q

        sample_norm = np.linalg.norm(sample.astype(np.float64), axis=1)
        global_stats["sample_quantiles_norm"] = {
            str(qi): float(np.quantile(sample_norm, qi)) for qi in [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
        }

    # Write JSON
    with open(out_dir / "global_action_stats.json", "w") as f:
        json.dump(global_stats, f, indent=2)
    print("✅ wrote global_action_stats.json")

    # Write per-episode CSV
    csv_path = out_dir / "episode_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "file", "steps",
            "zero_frac_norm", "exact_zero_frac_norm",
            "mean_norm", "median_norm", "p95_norm", "mean_abs_norm",
        ]
        # Per dof columns
        for name in dof_names:
            header += [f"zero_frac_{name}", f"exact_zero_frac_{name}", f"mean_abs_{name}", f"std_{name}", f"min_{name}", f"max_{name}"]
        w.writerow(header)

        for es in episode_stats:
            row = [
                es.file, es.steps,
                safe_float(es.zero_frac_norm), safe_float(es.exact_zero_frac_norm),
                safe_float(es.mean_norm), safe_float(es.median_norm), safe_float(es.p95_norm), safe_float(es.mean_abs_norm),
            ]
            for j in range(D):
                row += [
                    safe_float(es.zero_frac_dof[j]),
                    safe_float(es.exact_zero_frac_dof[j]),
                    safe_float(es.mean_abs_dof[j]),
                    safe_float(es.std_dof[j]),
                    safe_float(es.min_dof[j]),
                    safe_float(es.max_dof[j]),
                ]
            w.writerow(row)

    print("✅ wrote episode_stats.csv")

    # Rank worst episodes
    episode_stats_sorted = sorted(
        [es for es in episode_stats if es.steps > 0 and not math.isnan(es.zero_frac_norm)],
        key=lambda es: (-es.zero_frac_norm, es.mean_abs_norm)
    )
    top_k = min(int(args.top_k), len(episode_stats_sorted))
    worst = episode_stats_sorted[:top_k]

    with open(out_dir / "worst_episodes_by_zero_frac.txt", "w") as f:
        for es in worst:
            f.write(f"{es.zero_frac_norm:.4f}\t{es.mean_abs_norm:.6f}\t{es.steps}\t{es.file}\n")
    print("✅ wrote worst_episodes_by_zero_frac.txt")

    # Print a quick console summary
    print("\n--- GLOBAL SUMMARY ---")
    print(f"steps: {total_steps}")
    print(f"near-zero (norm < eps_norm): {global_stats['near_zero_fraction_norm']:.4f}")
    print(f"exact-zero (norm == 0):      {global_stats['exact_zero_fraction_norm']:.4f}")
    for j, name in enumerate(dof_names):
        nz = global_stats["near_zero_fraction_dof"][j]
        ez = global_stats["exact_zero_fraction_dof"][j]
        print(f"{name:>8s} | near0(abs<eps)={nz:.4f} | exact0={ez:.4f}")

    # ------------------ Plots from reservoir sample ------------------

    if sample.shape[0] >= 100:
        plots_dir = out_dir / "plots"
        ensure_out_dir(plots_dir)

        # Per-DoF histogram and log-abs hist
        for j, name in enumerate(dof_names):
            v = sample[:, j].astype(np.float64)
            # robust range for symmetric hist around 0
            lo, hi = np.quantile(v, [0.01, 0.99])
            r = max(abs(lo), abs(hi))
            if not np.isfinite(r) or r <= 0:
                r = float(np.max(np.abs(v))) if v.size else 1.0
            r = max(r * 1.2, eps * 10.0)  # ensure visible around eps
            save_histogram(
                values=v,
                title=f"Action distribution: {name}",
                xlabel=name,
                out_path=plots_dir / f"hist_{name}.png",
                bins=200,
                range_=(-r, r),
                logy=False,
            )
            save_histogram(
                values=v,
                title=f"Action distribution (log y): {name}",
                xlabel=name,
                out_path=plots_dir / f"hist_{name}_logy.png",
                bins=200,
                range_=(-r, r),
                logy=True,
            )
            save_logabs_hist(
                values=v,
                title=f"log10(|action|) distribution: {name}",
                out_path=plots_dir / f"logabs_{name}.png",
                bins=200,
            )

        # L2 norm histogram
        sample_norm = np.linalg.norm(sample.astype(np.float64), axis=1)
        hi = float(np.quantile(sample_norm, 0.99))
        hi = max(hi * 1.2, eps_norm * 10.0)
        save_histogram(
            values=sample_norm,
            title="Action L2 norm distribution",
            xlabel="||action||_2",
            out_path=plots_dir / "hist_norm.png",
            bins=200,
            range_=(0.0, hi),
            logy=False,
        )
        save_histogram(
            values=sample_norm,
            title="Action L2 norm distribution (log y)",
            xlabel="||action||_2",
            out_path=plots_dir / "hist_norm_logy.png",
            bins=200,
            range_=(0.0, hi),
            logy=True,
        )

        # Correlation heatmap
        save_corr_heatmap(sample, dof_names, plots_dir / "corr_heatmap.png")

        # Worst episodes bar plot
        if top_k > 0:
            # Short labels (file stem)
            labels = [Path(es.file).stem[:35] for es in worst]
            vals = [float(es.zero_frac_norm) for es in worst]
            save_episode_bar(
                values=vals,
                labels=labels,
                title=f"Top-{top_k} worst episodes by near-zero action (||a||_2 < {eps_norm:g})",
                ylabel="fraction near-zero steps",
                out_path=plots_dir / "worst_zero_frac_norm.png",
            )

            vals2 = [float(es.mean_abs_norm) for es in worst]
            save_episode_bar(
                values=vals2,
                labels=labels,
                title=f"Top-{top_k} worst episodes: mean(|a_i|) averaged over DoFs",
                ylabel="mean abs(action)",
                out_path=plots_dir / "worst_mean_abs.png",
            )

        print(f"✅ wrote plots -> {plots_dir}")

    # ------------------ Inspect: timeseries for specific files ------------------

    if args.inspect:
        inspect_dir = out_dir / "inspect"
        ensure_out_dir(inspect_dir)
        for p in args.inspect:
            fp = Path(p)
            if not fp.exists():
                print(f"[inspect] missing: {fp}")
                continue
            a = read_actions_from_tfrecord(fp, action_dim=D, batch_size=args.batch_size)
            if a.shape[0] == 0:
                print(f"[inspect] empty: {fp}")
                continue
            stem = fp.stem
            save_timeseries(a, dof_names, inspect_dir / f"{stem}_actions.png", title=f"Actions timeseries: {stem}")
            save_norm_timeseries(a, inspect_dir / f"{stem}_norm.png", title=f"||action||_2 timeseries: {stem}")
        print(f"✅ wrote inspect plots -> {inspect_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
