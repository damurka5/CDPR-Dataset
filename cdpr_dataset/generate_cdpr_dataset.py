#!/usr/bin/env python3
# Generate synthetic CDPR dataset (RLDS/Open-X style) without writing into VLA_CDPR.
# - Builds/uses a cached wrapper under cdpr_dataset/wrappers/ per (scene, objects) combo
# - Waypointed EE motion (up → above → down) with 3 cm tolerance
# - Simple non-overlap object placement near the workspace center
# - Unique per-episode output directories (no overwrites)

import os, sys, yaml, argparse, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

from cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation

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

def _wrapper_name(scene: str, object_names: list[str]) -> str:
    objs = "-".join(sorted(object_names))
    return f"{scene}__{objs}_wrapper.xml"

def build_wrapper_if_needed(scene_name: str,
                            object_names: list[str],
                            scene_z=-0.85,
                            ee_start=(0.0, 0.0, 0.25),
                            table_z=0.15,
                            settle_time=1.0) -> Path:
    """
    Compose a wrapper XML into our WRAP_DIR using the scene-switcher CLI.
    We cache per (scene, objects) combo; subsequent runs reuse that wrapper.
    """
    WRAP_DIR.mkdir(parents=True, exist_ok=True)
    wrapper_out = WRAP_DIR / _wrapper_name(scene_name, object_names)
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
    for obj in object_names:
        # placeholder XY; we re-place centrally in-process
        cmd += ["--object", f"{obj}:0.40,0.40,0.00"]

    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ Built wrapper: {wrapper_out}\n   Includes {len(object_names)} object(s).")
    return wrapper_out

def _episode_out_dir(wrapper_xml: Path, task_name: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{Path(wrapper_xml).stem}_{task_name}_{stamp}"
    return VIDEO_DIR / base

def run_episode(task_name: str, wrapper_xml: Path):
    """
    Run a single scripted episode on the given wrapper scene.
    Saves results only inside VIDEO_DIR/<unique>/...
    """
    sim = HeadlessCDPRSimulation(xml_path=str(wrapper_xml), output_dir=str(VIDEO_DIR))
    sim.initialize()

    # CENTRAL placement window: near (0,0) to encourage solid picks/pushes
    try:
        # tight ROI around origin (±12 cm); adjust if your table/objects need more space
        xy_bounds = ((-0.12, 0.12), (-0.12, 0.12), 0.10)
        place_objects_non_overlapping(sim, ["target_object"], xy_bounds, min_gap=0.015)
    except Exception as e:
        print("Object placement note:", e)

    if task_name == "pick_and_hover":
        script_pick_and_hover(sim, tol=0.03)
    elif "push" in task_name:
        # "left", "right", "forward", "back" inferred from name
        direction = "left" if "left" in task_name else ("right" if "right" in task_name
                    else ("forward" if "forward" in task_name else "back"))
        script_push(sim, direction=direction, tol=0.03)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    out_dir = _episode_out_dir(wrapper_xml, task_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim.save_trajectory_results(str(out_dir), f"{out_dir.name}")
    sim.cleanup()

def main():
    args = parse_args()
    ensure_dirs()

    scene_specs = []
    if args.catalog:
        cfg = load_catalog(args.catalog)
        defaults = cfg.get("defaults", {})
        scenes_cfg = cfg.get("scenes", [])
        for entry in scenes_cfg:
            if isinstance(entry, dict):
                scene_name = entry["name"]
                object_names = entry.get("objects", [])
            else:
                scene_name = str(entry); object_names = []
            scene_specs.append((scene_name, object_names, defaults))
    else:
        if args.scene is None or args.object is None:
            raise SystemExit("Provide --catalog or both --scene and --object.")
        scene_specs.append((args.scene, [args.object], {}))

    for scene_name, object_names, defaults in scene_specs:
        scene_z   = defaults.get("scene_z", -0.85)
        ee_start  = tuple(defaults.get("ee_start", (0.0, 0.0, 0.25)))
        table_z   = defaults.get("table_z", 0.15)
        settle_t  = defaults.get("settle_time", 1.0)

        wrapper_xml = build_wrapper_if_needed(scene_name, object_names,
                                              scene_z=scene_z,
                                              ee_start=ee_start,
                                              table_z=table_z,
                                              settle_time=settle_t)
        print(f"✅ Loaded scene '{scene_name}' with {len(object_names)} object(s). Wrapper at: {wrapper_xml}")
        for _ in range(args.episodes_per_scene):
            for t in args.tasks:
                run_episode(t, wrapper_xml)

if __name__ == "__main__":
    main()
