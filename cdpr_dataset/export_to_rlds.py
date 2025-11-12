#!/usr/bin/env python3
"""
Convert CDPR synthetic episodes under cdpr_dataset/datasets/cdpr_synth/videos/*
into RLDS TFRecords compatible with OpenVLA-OFT's LIBERO loaders.

Outputs:
- cdpr_dataset/datasets/cdpr_synth/tfrecords/cdpr_synth-{shard:05d}-of-{nshards:05d}.tfrecord
- cdpr_dataset/datasets/cdpr_synth/meta_dataset.json  (updated)
- cdpr_dataset/datasets/cdpr_synth/action_stats_cdpr_synth.json  (mean/std, min/max)

Assumptions:
- Each episode dir has: overview_video.mp4, ee_camera_video.mp4, trajectory_data.npz
- npz contains arrays for ee pose/targets and yaw/gripper per frame (we auto-detect keys)
- Frame counts of both videos match the control log length (we’ll trim to min length)

Install deps (same env as generator):
  pip install rlds tensorflow==2.15.* opencv-python-headless==4.* numpy==1.* tqdm
"""
import os, json, re, glob, math
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "datasets" / "cdpr_synth"
VIDEO_ROOT = DATASET_ROOT / "videos"
# TFREC_DIR  = DATASET_ROOT / "tfrecords"
META_PATH  = DATASET_ROOT / "meta_dataset.json"
# STATS_PATH = DATASET_ROOT / "action_stats_cdpr_synth.json"
DATASET_NAME = "libero_spatial_no_noops"   # to mirror LIBERO
TFREC_DIR = DATASET_ROOT / DATASET_NAME / "tfrecords"
STATS_PATH = DATASET_ROOT / f"action_stats_{DATASET_NAME}.json"

# --- Helpers -----------------------------------------------------------------

def _find(arrs, candidates):
    for k in candidates:
        if k in arrs: return k
    return None

def _npz_keys(npz):
    return [k for k in npz.files]

def _read_video_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    ok = cap.isOpened()
    while ok:
        ok, frame = cap.read()
        if not ok: break
        # BGR->RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def _delta(a):
    if len(a) < 2:
        return np.zeros_like(a)
    da = np.zeros_like(a)
    da[1:] = a[1:] - a[:-1]
    return da

def _safe_float(x):
    return float(x) if np.isfinite(x) else 0.0

def _episode_to_rlds(ep_dir: Path):
    """Return a list of RLDS steps (dicts) given an episode directory."""
    ov_path = ep_dir / "overview_video.mp4"
    ee_path = ep_dir / "ee_camera_video.mp4"
    npz_path = ep_dir / "trajectory_data.npz"
    if not (ov_path.exists() and ee_path.exists() and npz_path.exists()):
        raise FileNotFoundError(f"Missing files in {ep_dir}")

    # Load frames (trim to min length later)
    ov_frames = _read_video_frames(ov_path)
    ee_frames = _read_video_frames(ee_path)

    logs = np.load(npz_path, allow_pickle=True)
    keys = _npz_keys(logs)

    # Heuristics to locate arrays
    k_xyz = _find(logs, ["ee_xyz", "ee_pos", "ee_position", "target_xyz", "cmd_xyz"])
    k_yaw = _find(logs, ["ee_yaw", "yaw", "cmd_yaw"])
    k_grp = _find(logs, ["gripper", "gripper_open", "gripper_cmd", "gripper_pos"])
    if k_xyz is None:
        raise KeyError(f"Could not find EE XYZ in {npz_path} (keys={keys})")
    xyz = np.asarray(logs[k_xyz], dtype=np.float32)
    yaw = np.asarray(logs[k_yaw], dtype=np.float32) if k_yaw else np.zeros((len(xyz),), dtype=np.float32)
    if k_grp:
        gr  = np.asarray(logs[k_grp], dtype=np.float32)
        # Normalize to [0,1] if likely a binary command
        if gr.max() > 1.0: gr = (gr > 0.5).astype(np.float32)
    else:
        gr = np.zeros((len(xyz),), dtype=np.float32)

    # Construct 7D absolute action proxy [x,y,z, roll, pitch, yaw, gripper]
    # For CDPR we don't use roll/pitch -> zeros.
    # abs5 = np.zeros((len(xyz), 7), dtype=np.float32)
    # abs5[:, 0:3] = xyz
    # abs5[:, 3:5] = 0.0
    # abs5[:, 5]   = yaw
    # abs5[:, 6]   = gr
        # ----- Build 5D absolute signal [x,y,z,yaw,grip] -----
    abs5 = np.zeros((len(xyz), 5), dtype=np.float32)
    abs5[:, 0:3] = xyz
    abs5[:, 3]   = yaw
    abs5[:, 4]   = gr

    # Per-step deltas (training target)
    act = _delta(abs5)                          # shape [T,5]

    # Trim to common length across videos and logs
    T = min(len(ov_frames), len(ee_frames), len(act))
    ov_frames = ov_frames[:T]
    ee_frames = ee_frames[:T]
    act       = act[:T]
    state     = abs5[:T].astype(np.float32)     # proprio = 5D absolute

    # ----- Language from folder name -----
    m = re.match(r"(?P<scene>[^_]+)__([^-]+)-?wrapper_(?P<task>[a-z_]+)_\d{4}-", ep_dir.name)
    if m:
        scene = m.group("scene").replace("-", " ")
        task  = m.group("task").replace("_", " ")
        objm  = re.search(r"__([^_]+)_wrapper_", ep_dir.name)
        obj   = objm.group(1).replace("-", " ") if objm else "object"
        task_phrase = task.replace("pick and hover", "pick the object and lift it")
        lang = f"{task_phrase} for the {obj} on the {scene}."
    else:
        lang = "perform the task"

    # ----- Build RLDS steps -----
    steps = []
    for t in range(T):
        steps.append({
            "observation": {
                "full_image": ov_frames[t],     # HWC uint8
                "wrist_image": ee_frames[t],    # HWC uint8
                "state": state[t],              # float32[5]
                "task_description": lang,       # str
            },
            "action": act[t],                   # float32[5] (Δ)
            "is_terminal": (t == T-1),
            "is_first": (t == 0),
            "is_last": (t == T-1),
        })
    return steps, act


def _serialize_step(step):
    # Convert numpy + strings to tf.train.Example
    def _bytes_feature(b): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))
    def _float_feature(v): return tf.train.Feature(float_list=tf.train.FloatList(value=v))
    def _int_feature(v):   return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

    # encode images as PNG to keep TFRecords compact
    def _png_bytes(arr):
        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return buf.tobytes() if ok else b""

    obs = step["observation"]
    features = {
        # images
        # "observation/full_image": _bytes_feature(_png_bytes(obs["full_image"])),
        # "observation/wrist_image": _bytes_feature(_png_bytes(obs["wrist_image"])),
        "observation/primary": _bytes_feature(_png_bytes(obs["full_image"])),
        "observation/wrist":   _bytes_feature(_png_bytes(obs["wrist_image"])),
        # state (float vector length 5)
        "observation/state": _float_feature(obs["state"].astype(np.float32).tolist()),
        # language
        "observation/task_description": _bytes_feature(obs["task_description"].encode("utf-8")),
        # action
        "action": _float_feature(step["action"].astype(np.float32).tolist()),
        # flags
        "is_terminal": _int_feature([int(step["is_terminal"])]),
        "is_first": _int_feature([int(step["is_first"])]),
        "is_last": _int_feature([int(step["is_last"])]),
    }
    return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()

def main():
    TFREC_DIR.mkdir(parents=True, exist_ok=True)
    ep_dirs = sorted([p for p in VIDEO_ROOT.glob("*") if p.is_dir()])
    if not ep_dirs:
        raise SystemExit(f"No episode dirs found under {VIDEO_ROOT}")

    all_actions = []
    shard_size = 50  # steps per shard (small, your dataset is small)
    shard_idx = 0
    writer = None
    steps_in_shard = 0

    def _new_writer():
        nonlocal shard_idx, steps_in_shard
        if writer: writer.close()
        path = TFREC_DIR / f"{DATASET_NAME}-train-{shard_idx:05d}-of-00001.tfrecord"
        shard_idx += 1
        steps_in_shard = 0
        return tf.io.TFRecordWriter(str(path))

    writer = _new_writer()

    for ep in tqdm(ep_dirs, desc="Exporting episodes"):
        try:
            steps, act = _episode_to_rlds(ep)
        except Exception as e:
            print(f"[skip] {ep.name}: {e}")
            continue
        all_actions.append(act)

        for st in steps:
            if steps_in_shard >= shard_size:
                writer.close()
                writer = _new_writer()
            writer.write(_serialize_step(st))
            steps_in_shard += 1

    writer.close()

    # Compute action normalization stats for OFT (per-dim mean/std, min/max)
    if all_actions:
        A = np.concatenate(all_actions, axis=0)
        stats = {
            "key": "cdpr_synth",   # unnorm_key you’ll reference in training cfg
            "dim": A.shape[1],
            "mean": A.mean(axis=0).tolist(),
            "std":  (A.std(axis=0) + 1e-6).tolist(),
            "min":  A.min(axis=0).tolist(),
            "max":  A.max(axis=0).tolist(),
            "description": "Δ[x,y,z,roll,pitch,yaw,gripper] for CDPR synthetic dataset"
        }
        with open(STATS_PATH, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Wrote action stats → {STATS_PATH}")

    # meta (lightweight)
    meta = {
        "name": "cdpr_synth",
        "format": "rlds",
        "fields": {
            "images": ["observation/primary", "observation/wrist"],
            "state":  "observation/state",
            "language": "observation/task_description",
            "action": "action",
        },
        "unnorm_key": "cdpr_synth"
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Wrote meta → {META_PATH}")
    print(f"✅ TFRecords in {TFREC_DIR}")

if __name__ == "__main__":
    main()
