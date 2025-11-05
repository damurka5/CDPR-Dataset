#!/usr/bin/env python3
# Generate synthetic CDPR dataset (RLDS/Open-X style) without writing into VLA_CDPR.
# - Builds/uses a cached wrapper under cdpr_dataset/wrappers/
# - Waypointed EE motion (up → above → down) with 3 cm tolerance
# - Simple non-overlap object placement on the table plane

import os, sys, yaml, argparse, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

from cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation

# We only use the scene switcher CLI to *compose* a wrapper xml once.
# All outputs remain inside this repo (CDPR_Dataset).
from .synthetic_tasks import (
    script_pick_and_hover,
    script_push,
    place_objects_non_overlapping,
)

# ----- I/O roots (never write into VLA_CDPR) -----
HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "datasets" / "cdpr_synth"
NPZ_DIR   = DATASET_ROOT / "npz"
VIDEO_DIR = DATASET_ROOT / "videos"
TFREC_DIR = DATASET_ROOT / "tfrecords"
WRAP_DIR  = HERE / "wrappers"

def ensure_dirs():
    for d in [NPZ_DIR, VIDEO_DIR, TFREC_DIR, WRAP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def parse_args():
    ap = argparse.ArgumentParser(
        prog="Generate synthetic CDPR dataset (RLDS/Open-X style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--catalog", type=str, default=None, help="Path to scene/object YAML")
    ap.add_argument("--out", type=str, default=str(DATASET_ROOT), help="Output root directory")
    ap.add_argument("--episodes_per_scene", type=int, default=2)
    ap.add_argument("--tasks", nargs="+", default=["pick_and_hover"])
    ap.add_argument("--strict_objects", action="store_true", default=False)
    ap.add_argument("--reinit_each_episode", action="store_true", default=False)

    # Convenience flags (optional) if you don't want to use a catalog
    ap.add_argument("--scene", type=str, default=None)
    ap.add_argument("--object", type=str, default=None)
    return ap.parse_args()

def load_catalog(catalog_path: str):
    with open(catalog_path, "r") as f:
        return yaml.safe_load(f)

def build_wrapper_if_needed(scene_name: str,
                            object_names: list,
                            scene_z=-0.85,
                            ee_start=(0.0, 0.0, 0.25),
                            table_z=0.15,
                            settle_time=1.0) -> Path:
    """
    Compose a wrapper XML into our WRAP_DIR using the scene-switcher CLI.
    We do this ONCE per scene; subsequent runs reuse the cached wrapper.
    """
    wrapper_out = WRAP_DIR / f"{scene_name}_auto_wrapper.xml"
    if wrapper_out.exists():
        print(f"✅ Using cached wrapper: {wrapper_out}")
        return wrapper_out

    cmd = [
        sys.executable, "-m", "cdpr_mujoco.cdpr_scene_switcher",
        "--scene", scene_name,
        "--scene_z", str(scene_z),
        "--ee_start", ",".join(map(str, ee_start)),
        "--table_z", str(table_z),
        "--settle_time", str(settle_time),
        "--wrapper_out", str(wrapper_out),
        "--object_on_table",
        "--object_dynamic",
    ]
    # initial XY is a placeholder; we will reposition via non-overlap placer
    for obj in object_names:
        cmd += ["--object", f"{obj}:0.50,0.50,0.00"]

    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ Built wrapper: {wrapper_out}\n   Includes {len(object_names)} object(s).")
    return wrapper_out

def run_episode(task_name: str, wrapper_xml: Path):
    """
    Run a single scripted episode on the given wrapper scene.
    Saves results only inside VIDEO_DIR.
    """
    sim = HeadlessCDPRSimulation(xml_path=str(wrapper_xml), output_dir=str(VIDEO_DIR))
    sim.initialize()

    # Place the target body(ies) without overlap on a conservative table ROI.
    # Most wrappers name the main task object body "target_object".
    try:
        xy_bounds = ((-0.35, 0.35), (-0.35, 0.35), 0.10)  # (xmin,xmax), (ymin,ymax), fixed_z
        place_objects_non_overlapping(sim, ["target_object"], xy_bounds, min_gap=0.02)
    except Exception as e:
        print("Object placement note:", e)

    if task_name == "pick_and_hover":
        # scripts accept **kwargs, including tol
        script_pick_and_hover(sim, tol=0.03)
        traj_name = f"{Path(wrapper_xml).stem}_{task_name}"
    elif "push" in task_name:
        direction = "left" if "left" in task_name else ("forward" if "forward" in task_name else "right")
        script_push(sim, direction=direction, tol=0.03)
        traj_name = f"{Path(wrapper_xml).stem}_{task_name}"
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Save only into our dataset directory
    out_dir = os.path.join(VIDEO_DIR, traj_name)
    sim.save_trajectory_results(out_dir, traj_name)
    sim.cleanup()

def main():
    args = parse_args()
    ensure_dirs()

    # Resolve scenes/objects
    if args.catalog:
        cfg = load_catalog(args.catalog)
        scenes = [s["name"] if isinstance(s, dict) else s for s in cfg.get("scenes", [])]
        # grab some objects; fall back if catalog has none at top-level
        objects = cfg.get("objects", []) or ["ketchup", "milk", "orange_juice"]
    else:
        if args.scene is None or args.object is None:
            raise SystemExit("Provide --catalog or both --scene and --object.")
        scenes = [args.scene]
        objects = [args.object]

    for scene_name in scenes:
        wrapper_xml = build_wrapper_if_needed(scene_name, objects)
        print(f"✅ Loaded scene '{scene_name}' with {len(objects)} object(s). Wrapper at: {wrapper_xml}")
        for _ in range(args.episodes_per_scene):
            for t in args.tasks:
                run_episode(t, wrapper_xml)

if __name__ == "__main__":
    main()
