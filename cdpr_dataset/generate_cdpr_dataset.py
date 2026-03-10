#!/usr/bin/env python3
# Generate synthetic CDPR dataset (RLDS/Open-X style) without writing into VLA_CDPR.
# - Builds/uses a cached wrapper under cdpr_dataset/wrappers/ per (scene, objects) combo
# - Waypointed EE motion (up → above → down) with 3 cm tolerance
# - Simple non-overlap object placement near the workspace center
# - Unique per-episode output directories (no overwrites)

import os, sys, yaml, argparse, subprocess
from pathlib import Path
from datetime import datetime
import math, random
import numpy as np

from cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation

from .synthetic_tasks import (
    script_pick_and_hover,
    script_push,
    script_move_to_xy,
    script_put_into_bowl,
    object_centers,
    place_objects_non_overlapping,
    task_language
)

# ----- I/O roots (never write into VLA_CDPR) -----
HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "datasets" / "cdpr_synth"
NPZ_DIR   = DATASET_ROOT / "npz"
VIDEO_DIR = DATASET_ROOT / "videos"
TFREC_DIR = DATASET_ROOT / "tfrecords"
WRAP_DIR  = HERE / "wrappers"
MIN_EE_START_Z = 0.35

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

def _auto_detect_object_body(sim, preferred: str | None = None) -> str:
    import mujoco as mj
    m = sim.model

    def has_body(name: str) -> bool:
        return mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name) != -1

    # 1) preferred exact match (just in case the wrapper kept the raw name)
    if preferred and has_body(preferred):
        return preferred

    # 2) common wrapper names
    for candidate in ("object", "target_object"):
        if has_body(candidate):
            return candidate

    # 3) heuristic: prefer placed LIBERO objects (prefix p0_, p1_, ...)
    robot_prefixes = ("rotor_", "slider_", "ee_", "camera_", "yaw_frame", "ee_platform", "finger_")

    placed_candidates = []
    free_candidates   = []

    for bid in range(m.nbody):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, bid)
        if not name or name == "world":
            continue
        if any(name.startswith(pfx) for pfx in robot_prefixes):
            continue

        jn = m.body_jntnum[bid]
        ja = m.body_jntadr[bid]
        has_free = any(m.jnt_type[ja + k] == mj.mjtJoint.mjJNT_FREE for k in range(jn))

        if has_free:
            free_candidates.append(name)
            if name.startswith("p0_") or name.startswith("p1_"):
                placed_candidates.append(name)

    if placed_candidates:
        return placed_candidates[0]
    if free_candidates:
        return free_candidates[0]

    # fallback: helpful error
    sample = []
    for i in range(m.nbody):
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i)
        if nm:
            sample.append(nm)
    raise RuntimeError(
        "Could not auto-detect object body. "
        f"Available bodies: {', '.join(sample[:30])}{' ...' if len(sample) > 30 else ''}"
    )

def _discover_movable_object_bodies(sim) -> list[str]:
    import mujoco as mj

    m = sim.model
    robot_prefixes = ("rotor_", "slider_", "ee_", "camera_", "yaw_frame", "ee_platform", "finger_")
    bodies: list[str] = []
    for bid in range(m.nbody):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, bid)
        if not name or name == "world":
            continue
        if any(name.startswith(pfx) for pfx in robot_prefixes):
            continue
        jn = int(m.body_jntnum[bid])
        ja = int(m.body_jntadr[bid])
        has_free = any(m.jnt_type[ja + k] == mj.mjtJoint.mjJNT_FREE for k in range(jn))
        if has_free:
            bodies.append(name)
    return bodies

def build_wrapper_if_needed(scene_name: str,
                            object_names: list[str],
                            scene_z=-0.85,
                            ee_start=(0.0, 0.0, 0.35),
                            table_z=0.15,
                            settle_time=0.0,
                            wrapper_out: str | Path | None = None,
                            use_cache: bool = True) -> Path:
    """
    Compose a wrapper XML into WRAP_DIR (or explicit wrapper_out) using the
    scene-switcher CLI.

    Default behavior keeps a cache per (scene, objects) combo. For RL runs that
    need per-episode randomization and cleanup, pass use_cache=False and a unique
    wrapper_out path inside the dataset repo.
    """
    WRAP_DIR.mkdir(parents=True, exist_ok=True)
    if wrapper_out is None:
        wrapper_path = WRAP_DIR / _wrapper_name(scene_name, object_names)
    else:
        wrapper_path = Path(wrapper_out).expanduser().resolve()
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and wrapper_path.exists():
        print(f"✅ Using cached wrapper: {wrapper_path}")
        return wrapper_path

    if (not use_cache) and wrapper_path.exists():
        try:
            wrapper_path.unlink()
        except Exception:
            # Continue and let scene switcher attempt to overwrite.
            pass

    ee_start = np.asarray(ee_start, dtype=float).reshape(3)
    ee_start[2] = max(float(ee_start[2]), MIN_EE_START_Z)

    cmd = [
        sys.executable, "-m", "cdpr_mujoco.cdpr_scene_switcher",
        "--scene", scene_name,
        "--scene_z", str(scene_z),
        "--ee_start", ",".join(map(str, ee_start)),
        "--table_z", str(table_z),
        "--settle_time", str(settle_time),
        "--wrapper_out", str(wrapper_path),
        "--object_on_table",
        "--object_dynamic",
    ]
    ROT_X_BY_OBJECT = {
        "ketchup":      math.radians(90),
        "milk":         math.radians(90),
        "orange_juice": math.radians(90),
        "tomato_sauce": math.radians(90),
        "ycb_wood_block": math.radians(90),
    }  

    # ---------- NEW: placement logic with "avoid gripper" constraint ----------
    # ee_start is a function argument: (x, y, z)
    ee_x, ee_y = float(ee_start[0]), float(ee_start[1])
    MIN_EE_DIST = 0.10  # keep initial objects 10 cm away from EE in XY

    def far_from_ee(x, y, min_dist=MIN_EE_DIST):
        return (x - ee_x) ** 2 + (y - ee_y) ** 2 >= (min_dist ** 2)

    placements = {}  # obj_name -> (x, y, z)

    has_bowl = any(obj == "red_bowl" or "plate" in obj for obj in object_names) # change back to "bowl"
    # treat the first non-bowl as the "main" object
    main_objects = [obj for obj in object_names if not ("bowl" in obj)]

    # 1) Place non-bowl objects near the center, but not under the gripper
    for obj in main_objects:
        max_tries = 100
        placed = False
        for _ in range(max_tries):
            x = random.uniform(-0.25, 0.25)
            y = random.uniform(-0.25, 0.25)
            if far_from_ee(x, y):
                placements[obj] = (x, y, 0.0)
                placed = True
                break
        if not placed:
            # last resort: place somewhere in workspace but still away from EE
            while True:
                x = random.uniform(-0.4, 0.4)
                y = random.uniform(-0.4, 0.4)
                if far_from_ee(x, y):
                    placements[obj] = (x, y, 0.0)
                    break

    # 2) Place bowl(s) offset from the first main object, not under EE
    if has_bowl:
        # choose which name we treat as bowl – red_bowl, white_bowl, etc.
        bowl_names = [obj for obj in object_names if "plate" in obj] # change back to "bowl"
        # if LIBERO asset is 'red_bowl', that's what you'll have here
        for bowl_name in bowl_names:
            if main_objects:
                anchor = main_objects[0]
                ax, ay, _ = placements.get(anchor, (0.0, 0.0, 0.0))
                r_min, r_max = 0.18, 0.28  # 18–28 cm away from anchor
                max_tries = 100
                placed = False
                for _ in range(max_tries):
                    r = random.uniform(r_min, r_max)
                    theta = random.uniform(0.0, 2.0 * math.pi)
                    bx = ax + r * math.cos(theta)
                    by = ay + r * math.sin(theta)
                    # keep inside workspace and away from EE
                    if -0.4 <= bx <= 0.4 and -0.4 <= by <= 0.4 and far_from_ee(bx, by):
                        placements[bowl_name] = (bx, by, 0.0)
                        placed = True
                        break
                if not placed:
                    # fallback: random, but away from EE
                    while True:
                        bx = random.uniform(-0.35, 0.35)
                        by = random.uniform(-0.35, 0.35)
                        if far_from_ee(bx, by):
                            placements[bowl_name] = (bx, by, 0.0)
                            break
            else:
                # bowl only: place centrally but away from EE
                max_tries = 100
                placed = False
                for _ in range(max_tries):
                    bx = random.uniform(-0.25, 0.25)
                    by = random.uniform(-0.25, 0.25)
                    if far_from_ee(bx, by):
                        placements[bowl_name] = (bx, by, 0.0)
                        placed = True
                        break
                if not placed:
                    while True:
                        bx = random.uniform(-0.35, 0.35)
                        by = random.uniform(-0.35, 0.35)
                        if far_from_ee(bx, by):
                            placements[bowl_name] = (bx, by, 0.0)
                            break

    # 3) Build CLI --object flags from placements
    for obj in object_names:
        # default random placement if somehow not in placements dict, still respecting EE
        if obj not in placements:
            max_tries = 100
            placed = False
            for _ in range(max_tries):
                x = random.uniform(-0.4, 0.4)
                y = random.uniform(-0.4, 0.4)
                if far_from_ee(x, y):
                    placements[obj] = (x, y, 0.0)
                    placed = True
                    break
            if not placed:
                # absolute last resort: ignore constraint
                placements[obj] = (0.0, 0.0, 0.0)

        x, y, z = placements[obj]

        angle = ROT_X_BY_OBJECT.get(obj, 0.0)   # rotation about X
        half  = angle / 2.0
        qx = math.sin(half)
        qy = 0.0
        qz = 0.0
        qw = math.cos(half)
        cmd += [
            "--object",
            f"{obj}:{x:.3f},{y:.3f},{z:.3f}:{qx},{qy},{qz},{qw}"
        ]
        
    # subprocess.run(cmd, check=True)
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd)  # NOTE: no check=True

    if proc.returncode != 0:
        # On macOS, cdpr_scene_switcher may segfault after writing wrapper XML.
        # If the wrapper exists, we can safely continue.
        if wrapper_path.exists() and wrapper_path.stat().st_size > 0:
            print(
                f"⚠️ cdpr_scene_switcher exited with code {proc.returncode}, "
                f"but wrapper was created. Continuing with: {wrapper_path}"
            )
        else:
            raise RuntimeError(
                f"cdpr_scene_switcher failed (code {proc.returncode}) and wrapper was not created."
            )

    print(f"✅ Built wrapper: {wrapper_path}\n   Includes {len(object_names)} object(s).")
    return wrapper_path


def _episode_out_dir(wrapper_xml: Path, task_name: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{Path(wrapper_xml).stem}_{task_name}_{stamp}"
    return VIDEO_DIR / base

def run_episode(task_name: str, wrapper_xml: Path, catalog_object_name: str):
    sim = HeadlessCDPRSimulation(xml_path=str(wrapper_xml), output_dir=str(VIDEO_DIR))
    sim.initialize()

    real_obj = sim.get_object_body_name() or catalog_object_name
    movable_objects = _discover_movable_object_bodies(sim)
    if real_obj and real_obj not in movable_objects:
        movable_objects.append(real_obj)
    print(f"[run_episode] Using object body in model: {real_obj}")
    print(f"[run_episode] Placing objects: {movable_objects}")

    # CENTRAL placement window
    try:
        # z value is a hint; placement helper grounds each object on support surface.
        xy_bounds = ((-0.12, 0.12), (-0.12, 0.12), 0.0)
        place_objects_non_overlapping(sim, movable_objects, xy_bounds, min_gap=0.015)
    except Exception as e:
        print("Object placement note:", e)

    # >>> NEW: copy model body_quat -> freejoint qpos, so orientation from wrapper is used
    import mujoco as mj
    import numpy as np

    m, d = sim.model, sim.data
    for obj_name in movable_objects:
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, obj_name)
        if bid == -1:
            continue
        # body_quat is [w, x, y, z] from MJCF (includes cdpr_scene_switcher quat).
        body_q = m.body_quat[bid].copy()
        jn = m.body_jntnum[bid]
        ja = m.body_jntadr[bid]
        for k in range(jn):
            jid = ja + k
            if m.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
                qadr = m.jnt_qposadr[jid]
                d.qpos[qadr+3:qadr+7] = body_q  # [w,x,y,z]
                break

    mj.mj_forward(m, d)
    bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, real_obj)
    if bid != -1:
        print(f"[debug] world xquat for {real_obj} after sync = {d.xquat[bid]}")

    # Now recompute centers after orientation is applied
    cx, tz, _ = object_centers(sim, real_obj)
    ee0 = sim.get_end_effector_position().copy()
    print(f"[debug] EE0={ee0}")
    print(f"[debug] OBJ(AABB_center)=({cx[0]:.3f}, {cx[1]:.3f}, {tz:.3f})")
    print(f"[debug] dist_xy(EE0 -> OBJ_center) = {np.linalg.norm(cx - ee0[:2]):.3f} m")

    # After repositioning, recompute geometry
    cx, tz, _ = object_centers(sim, real_obj)

    # For comparison: MuJoCo body_xpos of that same body
    try:
        import mujoco as mj
        m, d = sim.model, sim.data
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, real_obj)
        body_pos = d.body_xpos[bid].copy()
    except Exception:
        body_pos = None

        # ---- Run the desired scripted task ----
    if task_name == "pick_and_hover":
        script_pick_and_hover(sim, object_body_name=real_obj, tol=0.015)

    elif "push" in task_name:
        direction = (
            "left" if "left" in task_name else
            "right" if "right" in task_name else
            "forward" if "forward" in task_name else
            "back"
        )
        script_push(sim, object_body_name=real_obj, direction=direction, tol=0.015)

    elif task_name == "move_to_center":
        goal_xy = (0.0, 0.0)
        script_move_to_xy(sim, object_body_name=real_obj, goal_xy=goal_xy, tol=0.015)

    elif task_name == "put_into_bowl":
        # here real_obj is the placed LIBERO body name of your object
        # and the bowl is 'red_bowl' (it will be resolved inside the script)
        from .synthetic_tasks import script_put_into_bowl  # or add to the import block above
        script_put_into_bowl(
            sim,
            object_body_name=catalog_object_name,   # use actual body name
            bowl_body_name="plate", # change back to "red_bowl"
            tol=0.015,
        )

    else:
        raise ValueError(f"Unknown task: {task_name}")

    # ---- Attach natural language prompt for this episode ----
    # catalog_object_name is what you passed into run_episode (e.g. 'milk')
    language = task_language(task_name, catalog_object_name)
    # store on the sim so save_summary can write it out
    setattr(sim, "language_instruction", language)

    out_dir = _episode_out_dir(wrapper_xml, "put_on_plate") # change back to task_name
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
        ee_start  = list(defaults.get("ee_start", (0.0, 0.0, 0.45)))
        ee_start[2] = max(float(ee_start[2]), MIN_EE_START_Z)
        table_z   = defaults.get("table_z", 0.15)
        settle_t  = defaults.get("settle_time", 0.0)
        # if any("bowl" in obj for obj in object_names):
        #     ee_start[2] = max(ee_start[2], 0.35)   # raise to at least 0.35 m

        # ee_start = tuple(ee_start)
        wrapper_xml = build_wrapper_if_needed(scene_name, object_names,
                                              scene_z=scene_z,
                                              ee_start=ee_start,
                                              table_z=table_z,
                                              settle_time=settle_t)
        print(f"✅ Loaded scene '{scene_name}' with {len(object_names)} object(s). Wrapper at: {wrapper_xml}")
        # Choose the (single) object name from your catalog; default to the first.
        obj_body = object_names[0] if object_names else "target_object"
        for _ in range(args.episodes_per_scene):
            for t in args.tasks:
                print(f"Using catalog object name: {obj_body}")
                run_episode(t, wrapper_xml, obj_body)


if __name__ == "__main__":
    main()
