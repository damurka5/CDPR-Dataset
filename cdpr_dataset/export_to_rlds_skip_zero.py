#!/usr/bin/env python3
"""
Export CDPR HUMAN_CONTROL episodes into RLDS-style TFRecords, with filtering of "zero" actions.

What this script does (vs. your original export_to_rlds.py):
  1) Builds steps exactly the same way (log-length is canonical; each camera stream resampled across full video span).
  2) Drops any step whose 5-DoF action is "zero" (all dims abs < eps), and drops the corresponding frames/state/timestamp.
  3) Re-writes is_first/is_last/is_terminal flags after filtering.
  4) Optionally keeps or drops the terminal step (which is zero in your delta-action construction).
  5) Skips exporting an episode if, after filtering, fewer than --min_steps remain.

Typical use:
  python export_to_rlds_skip_zero.py \
    --dataset_root /root/repo/cdpr_synth_10hz \
    --dataset_name libero_spatial_no_noops \
    --out_subdir tfrecords_human_control_fixed_nozeros \
    --eps 1e-4 --keep_terminal --min_steps 2

Notes:
  - "Zero action" is defined as: max(abs(action)) < eps (equivalently, all dims abs < eps).
  - If you want to remove ALL zeros including terminal: pass --drop_terminal.
"""

import os, json, re, argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm


# ---------------- Helpers ----------------

def _img_bytes(arr_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    return buf.tobytes() if ok else b""


def _npz_keys(npz) -> list[str]:
    return [k for k in npz.files]


def _read_video_frames(path: Path, desc: str = "") -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    pbar = tqdm(total=n if n > 0 else None, desc=desc, leave=False)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pbar.update(1)

    pbar.close()
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Decoded 0 frames from {path}")
    return frames


def _read_language_from_summary(summary_path: Path):
    if not summary_path.exists():
        return None
    try:
        for line in summary_path.read_text().splitlines():
            if line.strip().lower().startswith("language_instruction:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _fallback_language_from_folder(ep_dir: Path) -> str:
    m = re.match(r"(?P<prefix>[^_]+)_(?P<scene>[^_]+)__", ep_dir.name)
    if m:
        scene = m.group("scene").replace("-", " ")
        return f"perform the task on the {scene}."
    return "perform the task"


def _normalize_timestamps_to_seconds(ts: np.ndarray) -> np.ndarray:
    """
    Heuristic unit normalization: timestamps can be in s/ms/us/ns.
    Returns seconds.
    """
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    if ts.size < 2:
        return ts

    dt0 = float(np.median(np.diff(ts)))
    candidates = []
    for scale in (1.0, 1e3, 1e6, 1e9):
        dt = dt0 / scale
        if 1e-4 <= dt <= 1.0:
            score = abs(np.log(dt) - np.log(0.1))  # mild preference for ~10Hz
            candidates.append((score, scale))
    scale = min(candidates, key=lambda x: x[0])[1] if candidates else 1.0
    return ts / scale


def pick_video_indices_fullspan(ts_seconds: np.ndarray, num_video_frames: int) -> np.ndarray:
    """
    Map each log timestamp to a video frame index by normalized progress through the episode.

    This does not rely on FPS metadata. It spreads indices across [0 .. num_video_frames-1].
    """
    ts = np.asarray(ts_seconds, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if num_video_frames <= 0:
        raise ValueError("num_video_frames must be > 0")
    if num_video_frames == 1:
        return np.zeros((ts.size,), dtype=np.int64)

    t0 = float(ts[0])
    t1 = float(ts[-1])
    dur = t1 - t0

    if dur <= 1e-12:
        p = np.linspace(0.0, 1.0, ts.size, dtype=np.float64)
    else:
        p = (ts - t0) / dur
        p = np.clip(p, 0.0, 1.0)

    idx = np.rint(p * (num_video_frames - 1)).astype(np.int64)
    idx = np.clip(idx, 0, num_video_frames - 1)
    return idx


def _episode_to_steps(ep_dir: Path, verbose: bool = True):
    """
    Returns:
      steps: list[dict] length N
      act: np.ndarray (N,5) float32  (delta-action with last row = 0)
    """
    ov_path = ep_dir / "overview_video.mp4"
    ee_path = ep_dir / "ee_camera_video.mp4"
    npz_path = ep_dir / "trajectory_data.npz"
    summary_path = ep_dir / "summary.txt"

    if not (ov_path.exists() and ee_path.exists() and npz_path.exists()):
        raise FileNotFoundError(f"Missing files in {ep_dir}")

    # Decode videos (full decode, then subselect indices)
    ov_all = _read_video_frames(ov_path, desc=f"read overview {ep_dir.name}")
    ee_all = _read_video_frames(ee_path, desc=f"read wrist {ep_dir.name}")

    logs = np.load(npz_path, allow_pickle=True)
    keys = _npz_keys(logs)

    for req in ("timestamp", "ee_position", "control_signals"):
        if req not in logs:
            raise KeyError(f"Missing '{req}' in {npz_path}. Available keys: {keys}")

    ts_raw = np.asarray(logs["timestamp"])
    ts = _normalize_timestamps_to_seconds(ts_raw)

    xyz_all = np.asarray(logs["ee_position"], dtype=np.float32)
    ctrl_all = np.asarray(logs["control_signals"], dtype=np.float32)

    # Canonical length from logs
    N = int(min(len(ts), len(xyz_all), len(ctrl_all)))
    if N < 2:
        raise ValueError(f"Episode too short after alignment: N={N}")

    ts = ts[:N]
    xyz = xyz_all[:N]
    ctrl = ctrl_all[:N]

    ts_rel = (ts - ts[0]).astype(np.float32)

    if verbose:
        median_dt = float(np.median(np.diff(ts))) if N >= 2 else 0.0
        hz = (1.0 / median_dt) if median_dt > 1e-12 else float("nan")
        log_dur = float(ts[-1] - ts[0])
        print(f"logs: N={N}, duration={log_dur:.3f}s, median_dt={median_dt:.4f}s (~{hz:.2f}Hz)")
        print(f"videos: overview frames={len(ov_all)} | wrist frames={len(ee_all)}")

    # Resample videos across full span to match N
    ov_idx = pick_video_indices_fullspan(ts, num_video_frames=len(ov_all))
    ee_idx = pick_video_indices_fullspan(ts, num_video_frames=len(ee_all))
    ov_frames = [ov_all[i] for i in ov_idx]
    ee_frames = [ee_all[i] for i in ee_idx]

    # Build state + delta-action from logs (5D: x,y,z,yaw,grip)
    if ctrl.ndim != 2 or ctrl.shape[1] < 2:
        raise ValueError(f"control_signals has unexpected shape: {ctrl.shape}")

    yaw = ctrl[:, -2]
    gr  = ctrl[:, -1]

    abs5 = np.zeros((N, 5), np.float32)
    abs5[:, 0:3] = xyz
    abs5[:, 3] = yaw
    abs5[:, 4] = gr

    act = np.zeros_like(abs5)
    act[:-1] = abs5[1:] - abs5[:-1]
    dy = act[:-1, 3]
    act[:-1, 3] = (dy + np.pi) % (2*np.pi) - np.pi
    act[-1] = 0

    lang = _read_language_from_summary(summary_path) or _fallback_language_from_folder(ep_dir)

    steps = []
    for t in range(N):
        steps.append({
            "observation": {
                "full_image": ov_frames[t],
                "wrist_image": ee_frames[t],
                "state": abs5[t],
                "timestamp": float(ts_rel[t]),
                "task_description": lang,
            },
            "action": act[t],
            "is_terminal": (t == N - 1),
            "is_first": (t == 0),
            "is_last": (t == N - 1),
        })

    return steps, act


def _serialize_step(step):
    def _bytes_feature(b): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))
    def _float_feature(v): return tf.train.Feature(float_list=tf.train.FloatList(value=v))
    def _int_feature(v):   return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

    obs = step["observation"]
    features = {
        "observation/primary": _bytes_feature(_img_bytes(obs["full_image"])),
        "observation/wrist":   _bytes_feature(_img_bytes(obs["wrist_image"])),
        "observation/state": _float_feature(obs["state"].astype(np.float32).tolist()),
        "observation/timestamp": _float_feature([float(obs["timestamp"])]),
        "observation/task_description": _bytes_feature(obs["task_description"].encode("utf-8")),
        "action": _float_feature(step["action"].astype(np.float32).tolist()),
        "is_terminal": _int_feature([int(step["is_terminal"])]),
        "is_first": _int_feature([int(step["is_first"])]),
        "is_last": _int_feature([int(step["is_last"])]),
    }
    return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


def filter_zero_action_steps(
    steps: list[dict],
    act: np.ndarray,
    eps: float,
    keep_terminal: bool,
):
    """
    Drops steps where action is "zero" in all dims (abs < eps).
    If keep_terminal is True, the last step is always kept (even if action is zero).
    Returns filtered_steps, filtered_actions, stats dict.
    """
    act = np.asarray(act, dtype=np.float32)
    if act.ndim != 2 or act.shape[1] != 5:
        raise ValueError(f"Expected act shape (N,5), got {act.shape}")

    zero_mask = np.all(np.abs(act) < eps, axis=1)
    keep = ~zero_mask
    if keep_terminal and len(keep) > 0:
        keep[-1] = True  # keep final observation/terminal

    idx = np.nonzero(keep)[0].astype(int)
    filtered_steps = [steps[i] for i in idx]
    filtered_act = act[idx]

    # Recompute RLDS flags after filtering
    if len(filtered_steps) > 0:
        for i, st in enumerate(filtered_steps):
            st["is_first"] = (i == 0)
            st["is_last"] = (i == len(filtered_steps) - 1)
            st["is_terminal"] = st["is_last"]

    stats = {
        "N_before": int(len(steps)),
        "N_after": int(len(filtered_steps)),
        "num_zero_before": int(zero_mask.sum()),
        "frac_zero_before": float(zero_mask.mean()) if len(zero_mask) else 0.0,
        "kept_terminal": bool(keep_terminal),
    }
    return filtered_steps, filtered_act, stats


def compute_action_stats(all_actions: list[np.ndarray]) -> dict:
    """
    Simple per-dimension stats over concatenated actions.
    """
    if not all_actions:
        return {"count": 0}

    A = np.concatenate([np.asarray(a, np.float32) for a in all_actions], axis=0)
    stats = {
        "count": int(A.shape[0]),
        "mean": A.mean(axis=0).tolist(),
        "std":  A.std(axis=0).tolist(),
        "min":  A.min(axis=0).tolist(),
        "max":  A.max(axis=0).tolist(),
        "p01":  np.quantile(A, 0.01, axis=0).tolist(),
        "p99":  np.quantile(A, 0.99, axis=0).tolist(),
    }
    return stats


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True,
                    help="Root folder that contains the episodes directory (videos/ by default).")
    ap.add_argument("--episodes_subdir", type=str, default="videos",
                    help="Subdir under dataset_root that contains episode folders.")
    ap.add_argument("--dataset_name", type=str, default="libero_spatial_no_noops",
                    help="Name used for meta file and tfrecord naming.")
    ap.add_argument("--out_subdir", type=str, default="tfrecords_human_control_fixed_nozeros",
                    help="Output subdir under dataset_root/dataset_name/ where TFRecords are written.")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                    help="Split string used in the tfrecord filenames.")
    ap.add_argument("--eps", type=float, default=1e-4,
                    help="Zero-action threshold. A step is dropped if all |action_i| < eps.")
    ap.add_argument("--keep_terminal", action="store_true",
                    help="Keep the final step even if its action is zero (recommended).")
    ap.add_argument("--drop_terminal", action="store_true",
                    help="Drop the final step if its action is zero (removes ALL zeros, but last obs isn't final).")
    ap.add_argument("--min_steps", type=int, default=2,
                    help="Skip exporting an episode if fewer than this many steps remain after filtering.")
    ap.add_argument("--write_meta", action="store_true",
                    help="Also write meta_dataset.json to dataset_root/meta_dataset.json.")
    ap.add_argument("--meta_path", type=str, default=None,
                    help="Override meta json output path (default: dataset_root/meta_dataset.json).")
    ap.add_argument("--write_action_stats", action="store_true",
                    help="Write per-dimension action stats json under dataset_root/action_stats_<dataset_name>.json.")
    ap.add_argument("--verbose", action="store_true", help="Print per-episode debug.")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    video_root = dataset_root / args.episodes_subdir

    # output directory: dataset_root/<dataset_name>/<out_subdir>
    out_dir = dataset_root / args.dataset_name / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_dirs = sorted([p for p in video_root.glob("*") if p.is_dir()])
    if not ep_dirs:
        raise SystemExit(f"No episode dirs found under {video_root}")

    keep_terminal = bool(args.keep_terminal) and (not bool(args.drop_terminal))
    if args.drop_terminal:
        keep_terminal = False

    all_actions_kept = []
    per_episode_filter_stats = {}

    num_written = 0
    num_skipped = 0

    for ep in tqdm(ep_dirs, desc="Exporting episodes (skip zero actions)"):
        try:
            if args.verbose:
                print(f"\n==> {ep.name}")
            steps, act = _episode_to_steps(ep, verbose=args.verbose)

            steps_f, act_f, st = filter_zero_action_steps(
                steps, act, eps=float(args.eps), keep_terminal=keep_terminal
            )

            per_episode_filter_stats[ep.name] = st

            if len(steps_f) < int(args.min_steps):
                num_skipped += 1
                if args.verbose:
                    print(f"[skip] {ep.name}: only {len(steps_f)} steps after filtering (min_steps={args.min_steps}). "
                          f"(zero before: {st['num_zero_before']}/{st['N_before']})")
                continue

        except Exception as e:
            num_skipped += 1
            if args.verbose:
                print(f"[skip] {ep.name}: {e}")
            continue

        tfrec_path = out_dir / f"{args.dataset_name}-{args.split}-{ep.name}.tfrecord"
        with tf.io.TFRecordWriter(str(tfrec_path)) as w:
            for stp in steps_f:
                w.write(_serialize_step(stp))

        all_actions_kept.append(act_f)
        num_written += 1

    # Optional meta json
    if args.write_meta:
        meta = {
            "name": args.dataset_name,
            "format": "rlds",
            "fields": {
                "images": ["observation/primary", "observation/wrist"],
                "state":  "observation/state",
                "timestamp": "observation/timestamp",
                "language": "observation/task_description",
                "action": "action",
            },
            "unnorm_key": args.dataset_name,
        }
        meta_path = Path(args.meta_path) if args.meta_path else (dataset_root / "meta_dataset.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"✅ Wrote meta → {meta_path}")

    # Optional action stats
    if args.write_action_stats:
        stats = compute_action_stats(all_actions_kept)
        stats["eps"] = float(args.eps)
        stats["keep_terminal"] = bool(keep_terminal)
        stats["episodes_written"] = int(num_written)
        stats["episodes_skipped"] = int(num_skipped)

        stats_path = dataset_root / f"action_stats_{args.dataset_name}.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Wrote action stats → {stats_path}")

    # Always write per-episode filter stats (small and useful for debugging)
    filter_stats_path = out_dir / "filter_stats_per_episode.json"
    with open(filter_stats_path, "w") as f:
        json.dump(per_episode_filter_stats, f, indent=2)
    print(f"✅ Wrote per-episode filter stats → {filter_stats_path}")

    print(f"✅ Episode TFRecords in {out_dir}")
    print(f"Summary: written={num_written} | skipped={num_skipped} | eps={args.eps} | keep_terminal={keep_terminal}")


if __name__ == "__main__":
    main()
