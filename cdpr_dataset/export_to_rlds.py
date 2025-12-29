#!/usr/bin/env python3
"""
Convert CDPR HUMAN_CONTROL episodes into RLDS TFRecords compatible with OpenVLA-OFT's LIBERO loaders.

Writes:
- One TFRecord per episode folder
- meta_dataset.json (updated)
- action_stats_<DATASET_NAME>.json

Assumptions:
- Each episode dir has: overview_video.mp4, ee_camera_video.mp4, trajectory_data.npz
- summary.txt contains 'language_instruction: ...' (optional but preferred)
"""

import os, json, re
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "datasets" / "cdpr_synth"

# === CHANGE: read from HUMAN_CONTROL instead of videos ===
VIDEO_ROOT = DATASET_ROOT / "HUMAN_CONTROL"

# Keep same structure requested by OpenVLA-OFT / LIBERO-style
DATASET_NAME = "libero_spatial_no_noops"
TFREC_DIR   = DATASET_ROOT / DATASET_NAME / "tfrecords_human_control"
META_PATH   = DATASET_ROOT / "meta_dataset.json"
STATS_PATH  = DATASET_ROOT / f"action_stats_{DATASET_NAME}.json"


# ---------------- Helpers ----------------

def _find(arrs, candidates):
    for k in candidates:
        if k in arrs:
            return k
    return None

def _img_bytes(arr):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes() if ok else b""

def _npz_keys(npz):
    return [k for k in npz.files]

def _read_video_frames(path: Path, desc: str = ""):
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


def _delta(a: np.ndarray):
    if len(a) < 2:
        return np.zeros_like(a)
    da = np.zeros_like(a)
    da[1:] = a[1:] - a[:-1]
    return da

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

def _fallback_language_from_folder(ep_dir: Path):
    # Your old folder parsing fallback, kept in case summary.txt missing
    m = re.match(r"(?P<prefix>[^_]+)_(?P<scene>[^_]+)__", ep_dir.name)
    if m:
        scene = m.group("scene").replace("-", " ")
        return f"perform the task on the {scene}."
    return "perform the task"

def _episode_to_rlds(ep_dir: Path):
    ov_path = ep_dir / "overview_video.mp4"
    ee_path = ep_dir / "ee_camera_video.mp4"
    npz_path = ep_dir / "trajectory_data.npz"
    summary_path = ep_dir / "summary.txt"

    if not (ov_path.exists() and ee_path.exists() and npz_path.exists()):
        raise FileNotFoundError(f"Missing files in {ep_dir}")

    ov_frames = _read_video_frames(ov_path, desc=f"read overview {ep_dir.name}")
    ee_frames = _read_video_frames(ee_path, desc=f"read wrist {ep_dir.name}")

    logs = np.load(npz_path, allow_pickle=True)
    keys = _npz_keys(logs)

    k_xyz = _find(logs, ["ee_xyz", "ee_pos", "ee_position", "target_xyz", "cmd_xyz"])
    k_yaw = _find(logs, ["ee_yaw", "yaw", "cmd_yaw"])
    k_grp = _find(logs, ["gripper", "gripper_open", "gripper_cmd", "gripper_pos"])

    if k_xyz is None:
        raise KeyError(f"Could not find EE XYZ in {npz_path} (keys={keys})")

    xyz = np.asarray(logs[k_xyz], dtype=np.float32)
    yaw = np.asarray(logs[k_yaw], dtype=np.float32) if k_yaw else np.zeros((len(xyz),), dtype=np.float32)

    if k_grp:
        gr = np.asarray(logs[k_grp], dtype=np.float32)
        if gr.max() > 1.0:
            gr = (gr > 0.5).astype(np.float32)
    else:
        gr = np.zeros((len(xyz),), dtype=np.float32)

    # 5D absolute state [x,y,z,yaw,grip]
    abs5 = np.zeros((len(xyz), 5), dtype=np.float32)
    abs5[:, 0:3] = xyz
    abs5[:, 3]   = yaw
    abs5[:, 4]   = gr

    # Training target: delta action
    act = _delta(abs5)

    # Trim to common length
    T = min(len(ov_frames), len(ee_frames), len(act))
    ov_frames = ov_frames[:T]
    ee_frames = ee_frames[:T]
    act       = act[:T]
    state     = abs5[:T].astype(np.float32)

    # Prefer summary.txt language instruction
    lang = _read_language_from_summary(summary_path) or _fallback_language_from_folder(ep_dir)

    steps = []
    for t in range(T):
        steps.append({
            "observation": {
                "full_image": ov_frames[t],
                "wrist_image": ee_frames[t],
                "state": state[t],
                "task_description": lang,
            },
            "action": act[t],
            "is_terminal": (t == T - 1),
            "is_first": (t == 0),
            "is_last": (t == T - 1),
        })
    return steps, act

def _serialize_step(step):
    def _bytes_feature(b): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))
    def _float_feature(v): return tf.train.Feature(float_list=tf.train.FloatList(value=v))
    def _int_feature(v):   return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

    def _png_bytes(arr):
        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return buf.tobytes() if ok else b""

    obs = step["observation"]
    features = {
        "observation/primary": _bytes_feature(_img_bytes(obs["full_image"])),
        "observation/wrist":   _bytes_feature(_img_bytes(obs["wrist_image"])),
        "observation/state": _float_feature(obs["state"].astype(np.float32).tolist()),
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

        # === CHANGE: one TFRecord per episode ===
        tfrec_path = TFREC_DIR / f"{DATASET_NAME}-train-{ep.name}.tfrecord"
        with tf.io.TFRecordWriter(str(tfrec_path)) as w:
            for st in steps:
                w.write(_serialize_step(st))

        all_actions.append(act)

    # Stats across all steps
    if all_actions:
        A = np.concatenate(all_actions, axis=0)
        stats = {
            "key": "cdpr_human_control",
            "dim": A.shape[1],
            "mean": A.mean(axis=0).tolist(),
            "std":  (A.std(axis=0) + 1e-6).tolist(),
            "min":  A.min(axis=0).tolist(),
            "max":  A.max(axis=0).tolist(),
            "description": "Δ[x,y,z,yaw,gripper] for CDPR HUMAN_CONTROL dataset"
        }
        with open(STATS_PATH, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Wrote action stats → {STATS_PATH}")

    meta = {
        "name": "cdpr_human_control",
        "format": "rlds",
        "fields": {
            "images": ["observation/primary", "observation/wrist"],
            "state":  "observation/state",
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
