#!/usr/bin/env python3
import os, sys, json, shutil, yaml, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

# robot package (installed from your VLA_CDPR)
from cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation
# re-use switcher discovery so we match your package’s expectations
from cdpr_mujoco.cdpr_scene_switcher import find_scene_xml, find_object_xml

from .synthetic_tasks import (
    XYZ_BOUNDS, GRIP_RANGE,
    script_pick_and_hover, script_push
)

HERE = Path(__file__).resolve().parent

# ---------- utilities ----------
def run_cmd(cmd):
    print(">>", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def filter_existing_objects(obj_names):
    """Keep only objects that resolve via find_object_xml; warn for the rest."""
    ok, missing = [], []
    for name in obj_names:
        try:
            _ = find_object_xml(name)
            ok.append(name)
        except FileNotFoundError:
            missing.append(name)
    if missing:
        print(f"[warn] skipping {len(missing)} missing object(s): {', '.join(missing)}")
    return ok

# ---------- Episode buffer ----------
class EpisodeBuffer:
    def __init__(self, meta):
        self.meta = meta
        self.steps = []

    def add(self, obs, act_abs7, act_delta7, is_first, is_last):
        self.steps.append({
            "observations/full_image": obs["full_image"],
            "observations/wrist_image": obs["wrist_image"],
            "observations/state": obs["state"].astype(np.float32),
            "language_instruction": obs["task_description"],
            "action/abs_7": act_abs7.astype(np.float32),
            "action/delta_7": act_delta7.astype(np.float32),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_last),
            "discount": np.float32(1.0),
            "reward": np.float32(0.0),
        })

    def save_npz(self, out_dir: Path, ep_name: str):
        out = ensure_dir(out_dir) / f"{ep_name}.npz"
        np.savez_compressed(out, meta=json.dumps(self.meta), steps=self.steps)
        return out

# ---------- optional RLDS exporter ----------
def try_export_rlds_tfrecord(out_dir: Path, ep_buffers):
    try:
        import tensorflow as tf
        import rlds  # noqa: F401
        from PIL import Image
        import io
    except Exception:
        print("[export] tensorflow/rlds not available; skipping TFRecord export.")
        return None

    tf_out = ensure_dir(out_dir / "tfrecords")
    writer = tf.io.TFRecordWriter(str(tf_out / "cdpr_synth.tfrecord"))
    print("[export] Writing RLDS TFRecord …")

    def _bytes_feature(x): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
    def _float_list(x):    return tf.train.Feature(float_list=tf.train.FloatList(value=list(map(float, x))))
    def _int64(v):         return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))
    def _png_bytes(arr):
        im = Image.fromarray(arr)
        buf = io.BytesIO()
        im.save(buf, format="PNG");  return buf.getvalue()

    for ep in ep_buffers:
        for st in ep.steps:
            feat = {
                "is_first":    _int64(st["is_first"]),
                "is_last":     _int64(st["is_last"]),
                "is_terminal": _int64(st["is_terminal"]),
                "discount":    tf.train.Feature(float_list=tf.train.FloatList(value=[float(st["discount"])])),
                "reward":      tf.train.Feature(float_list=tf.train.FloatList(value=[float(st["reward"])])),
                "observations/full_image_png":  _bytes_feature(_png_bytes(st["observations/full_image"])),
                "observations/wrist_image_png": _bytes_feature(_png_bytes(st["observations/wrist_image"])),
                "observations/state":           _float_list(st["observations/state"]),
                "language_instruction": _bytes_feature(st["language_instruction"].encode("utf-8")),
                "action/abs_7":   _float_list(st["action/abs_7"]),
                "action/delta_7": _float_list(st["action/delta_7"]),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(example.SerializeToString())
    writer.close()
    return tf_out

# ---------- Wrapper builder (now robust to missing objects) ----------
def filter_existing_scene(scene_name):
    try:
        _ = find_scene_xml(scene_name)  # will raise if missing
        return scene_name
    except FileNotFoundError:
        print(f"[warn] skipping missing scene: {scene_name}")
        return None

def build_wrapper(scene, objects, wrapper_out, scene_z, ee_start, table_z, dynamic=True, settle=1.0):
    scene_ok = filter_existing_scene(scene)
    if scene_ok is None:
        return None

    objects_ok = filter_existing_objects(objects)
    if not objects_ok:
        print(f"[warn] No valid objects for scene '{scene}'. Skipping wrapper.")
        return None

    cmd = [
        sys.executable, "-m", "cdpr_mujoco.cdpr_scene_switcher",
        "--scene", scene_ok,
        "--scene_z", str(scene_z),
        "--ee_start", ",".join(map(str, ee_start)),
        "--table_z", str(table_z),
        "--settle_time", str(settle),
        "--wrapper_out", str(wrapper_out),
        "--object_on_table",
    ]
    if dynamic:
        cmd.append("--object_dynamic")
    for ob in objects_ok:
        cmd += ["--object", f"{ob}:0.50,0.50,0.00"]
    run_cmd(cmd)
    return wrapper_out

# ---------- Main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser("Generate synthetic CDPR dataset (RLDS/Open-X style)")
    ap.add_argument("--catalog", default=str(HERE/"datasets"/"cdpr_scene_catalog.yaml"))
    ap.add_argument("--out", default=str(HERE/"datasets"/"cdpr_synth"))
    ap.add_argument("--episodes_per_scene", type=int, default=5)
    ap.add_argument("--tasks", nargs="+", default=["pick_and_hover", "push_left", "push_forward"])
    ap.add_argument("--strict_objects", action="store_true",
                    help="If set, error out if any listed object is missing.")
    ap.add_argument("--reinit_each_episode", action="store_true",
                help="Re-initialize the simulator before each episode (ensures clean recording).")

    args = ap.parse_args()

    out_root = ensure_dir(Path(args.out))
    cats = yaml.safe_load(Path(args.catalog).read_text())
    defaults = cats.get("defaults", {})
    scenes = cats["scenes"]

    ep_buffers = []
    ep_counter = 0

    for sc in scenes:
        scene = sc["name"]
        objects = sc["objects"][:]
        scene_z = sc.get("scene_z", defaults.get("scene_z", -0.85))
        ee_start = sc.get("ee_start", defaults.get("ee_start", [0,0,0.25]))
        table_z  = sc.get("table_z", defaults.get("table_z", 0.15))
        dynamic  = bool(sc.get("object_dynamic", defaults.get("object_dynamic", True)))
        settle   = float(sc.get("settle_time", defaults.get("settle_time", 1.0)))

        wrapper = Path(HERE/"wrappers"/f"{scene}_auto_wrapper.xml")
        wrapper.parent.mkdir(parents=True, exist_ok=True)

        # strict mode check
        if args.strict_objects:
            # will raise if any single object is missing
            for ob in objects:
                _ = find_object_xml(ob)

        wrapper_path = build_wrapper(scene, objects, wrapper, scene_z, ee_start, table_z, dynamic, settle)
        if wrapper_path is None or not wrapper.exists():
            print(f"[warn] skipping scene '{scene}' (no wrapper produced).")
            continue

        # init sim
        sim = HeadlessCDPRSimulation(str(wrapper), output_dir=str(out_root/"videos"))
        sim.initialize()
        if hasattr(sim, "hold_current_pose"):
            sim.hold_current_pose(warm_steps=0)

        # episodes
        for k in range(args.episodes_per_scene):
            # (optional) ensure clean recorder state
            if args.reinit_each_episode and k > 0:
                sim.cleanup()
                sim = HeadlessCDPRSimulation(str(wrapper), output_dir=str(out_root/"videos"))
                sim.initialize()
                if hasattr(sim, "hold_current_pose"):
                    sim.hold_current_pose(warm_steps=0)

            task_name = args.tasks[k % len(args.tasks)]
            if task_name == "pick_and_hover":
                logs = script_pick_and_hover(sim, task_text=f"Pick up the {objects[0]}, then hover.")
            elif task_name == "push_left":
                logs = script_push(sim, direction="left",  task_text=f"Push the {objects[0]} to the left.")
            elif task_name == "push_forward":
                logs = script_push(sim, direction="forward", task_text=f"Push the {objects[0]} forward.")
            else:
                logs = script_pick_and_hover(sim, task_text=f"Pick up the {objects[0]}, then hover.")

            meta = {
                "scene": scene,
                "objects": objects,
                "task": task_name,
                "wrapper_xml": str(wrapper),
                "xyz_bounds": XYZ_BOUNDS,
                "grip_range": GRIP_RANGE,
                "episode_index": ep_counter,
                "time": datetime.now().isoformat(),
                "axes": {"+Y": "overview camera forward", "+X": "right"},
            }
            buf = EpisodeBuffer(meta=meta)
            for i, (obs, a_abs7, a_d7) in enumerate(logs):
                buf.add(obs, a_abs7, a_d7, is_first=(i==0), is_last=(i==len(logs)-1))

            ep_name = f"{scene}_{task_name}_{ep_counter:05d}"
            # 1) save NPZ
            buf.save_npz(out_root/"npz", ep_name)

            # 2) save per-episode videos + trajectory dump
            ep_vid_dir = ensure_dir(out_root/"videos"/ep_name)
            try:
                sim.save_trajectory_results(str(ep_vid_dir), ep_name)
            except Exception as e:
                print(f"[warn] video save failed for {ep_name}: {e}")

            # 3) clear recorders for next episode (best-effort)
            cleared = False
            for attr in ("reset_recorders", "clear_recorders", "reset_video_buffers", "clear_frame_buffers"):
                if hasattr(sim, attr):
                    try:
                        getattr(sim, attr)()
                        cleared = True
                        break
                    except Exception:
                        pass
            if not cleared and not args.reinit_each_episode:
                # If your sim doesn't expose a reset, recommend the safe mode
                print("[hint] If videos include frames from previous episodes, re-run with --reinit_each_episode")

            ep_buffers.append(buf)
            ep_counter += 1


        sim.cleanup()

    # Optional TFRecord export
    try_export_rlds_tfrecord(out_root, ep_buffers)

    # Dataset meta
    (out_root/"meta_dataset.json").write_text(json.dumps({
        "num_episodes": ep_counter,
        "scenes": [s["name"] for s in scenes],
        "tasks": args.tasks,
        "schema": {
            "observations": ["full_image (HWC uint8)", "wrist_image (HWC uint8)", "state (8D normalized)"],
            "actions": ["action/abs_7", "action/delta_7"],
            "flags": ["is_first", "is_last", "is_terminal"],
            "scalars": ["discount", "reward"],
            "language": "language_instruction",
        },
        "bounds": {"xyz": XYZ_BOUNDS, "gripper": GRIP_RANGE},
        "note": "Yaw/pitch/roll follow your CDPR (only yaw used)."
    }, indent=2))

if __name__ == "__main__":
    main()
