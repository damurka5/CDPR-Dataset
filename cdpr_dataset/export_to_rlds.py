#!/usr/bin/env python3
"""
Convert CDPR HUMAN_CONTROL episodes into RLDS TFRecords compatible with OpenVLA-OFT's LIBERO loaders.

This version fixes camera/log alignment issues by:
  - Using the log trajectory length (timestamps/target_position/control_signals) as the canonical episode length.
  - Resampling EACH camera stream independently to that length.
  - Sampling frames across the FULL video span (start->end), i.e. "keep full video but skip frames",
    instead of relying on (often-wrong) FPS metadata.

It also writes per-step timestamps (seconds since episode start) into the TFRecord as:
  observation/timestamp

Assumptions per episode directory:
  - overview_video.mp4
  - ee_camera_video.mp4
  - trajectory_data.npz
  - (optional) summary.txt with "language_instruction: ..."
"""

import os, json, re
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATASET_ROOT = "/root/repo/cdpr_synth_10hz/"

VIDEO_ROOT = Path(DATASET_ROOT + "videos/")

DATASET_NAME = "libero_spatial_no_noops/"
TFREC_DIR   = Path(DATASET_ROOT + DATASET_NAME + "tfrecords_human_control_fixed")
META_PATH   = DATASET_ROOT + "meta_dataset.json"
STATS_PATH  = DATASET_ROOT + f"action_stats_{DATASET_NAME}.json"


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

    This *does not* rely on FPS metadata. It guarantees we spread indices across the full
    [0 .. num_video_frames-1] range (i.e., "full video but skipped frames").
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
        # Degenerate timestamps -> uniform index ramp
        p = np.linspace(0.0, 1.0, ts.size, dtype=np.float64)
    else:
        p = (ts - t0) / dur
        p = np.clip(p, 0.0, 1.0)

    idx = np.rint(p * (num_video_frames - 1)).astype(np.int64)
    idx = np.clip(idx, 0, num_video_frames - 1)
    return idx

def _episode_to_rlds(ep_dir: Path):
    ov_path = ep_dir / "overview_video.mp4"
    ee_path = ep_dir / "ee_camera_video.mp4"
    npz_path = ep_dir / "trajectory_data.npz"
    summary_path = ep_dir / "summary.txt"

    if not (ov_path.exists() and ee_path.exists() and npz_path.exists()):
        raise FileNotFoundError(f"Missing files in {ep_dir}")

    # Decode videos
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

    median_dt = float(np.median(np.diff(ts))) if N >= 2 else 0.0
    hz = (1.0 / median_dt) if median_dt > 1e-12 else float("nan")
    log_dur = float(ts[-1] - ts[0])

    print(f"logs: N={N}, duration={log_dur:.3f}s, median_dt={median_dt:.4f}s (~{hz:.2f}Hz)")
    print(f"videos: overview frames={len(ov_all)} | wrist frames={len(ee_all)}")

    # --- Resample videos across their full span to match N ---
    ov_idx = pick_video_indices_fullspan(ts, num_video_frames=len(ov_all))
    ee_idx = pick_video_indices_fullspan(ts, num_video_frames=len(ee_all))

    # Materialize resampled frames
    ov_frames = [ov_all[i] for i in ov_idx]
    ee_frames = [ee_all[i] for i in ee_idx]

    # --- Build state + delta-action from logs (5D: x,y,z,yaw,grip) ---
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
                "timestamp": float(ts_rel[t]),  # seconds since episode start
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


# ---------------- Main ----------------

def main():
    TFREC_DIR.mkdir(parents=True, exist_ok=True)

    ep_dirs = sorted([p for p in VIDEO_ROOT.glob("*") if p.is_dir()])
    if not ep_dirs:
        raise SystemExit(f"No episode dirs found under {VIDEO_ROOT}")

    all_actions = []

    for ep in tqdm(ep_dirs, desc="Exporting HUMAN_CONTROL episodes"):
        try:
            print(f"\n==> {ep.name}")
            steps, act = _episode_to_rlds(ep)
        except Exception as e:
            print(f"[skip] {ep.name}: {e}")
            continue

        tfrec_path = TFREC_DIR / f"libero_spatial_no_noops-train-{ep.name}.tfrecord"
        with tf.io.TFRecordWriter(str(tfrec_path)) as w:
            for st in steps:
                w.write(_serialize_step(st))

        all_actions.append(act)

    meta = {
        "name": "cdpr_human_control",
        "format": "rlds",
        "fields": {
            "images": ["observation/primary", "observation/wrist"],
            "state":  "observation/state",
            "timestamp": "observation/timestamp",
            "language": "observation/task_description",
            "action": "action",
        },
        "unnorm_key": "cdpr_human_control",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Wrote meta → {META_PATH}")
    print(f"✅ Episode TFRecords in {TFREC_DIR}")

if __name__ == "__main__":
    main()