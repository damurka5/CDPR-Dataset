#!/usr/bin/env python3
"""
Teleop data collection for CDPR on local machines (macOS/Linux).

Key features:
- Works even if wrapper XML was generated on another machine and contains absolute
  <include file="/root/repo/..."> paths. We patch them to local wrapper directory.
- Uses pynput for keyboard control.

Controls:
  - W/S : +Y / -Y
  - A/D : -X / +X
  - Arrow Up/Down : +Z / -Z
  - [ / ] : yaw - / +
  - X : open gripper
  - C : close gripper
  - Q or ESC : end episode and save

Output:
  cdpr_dataset/datasets/cdpr_synth/videos/<episode_dir>/{trajectory_data.npz, *.mp4, summary.txt}
"""

import argparse
import math
import mujoco as mj
import re
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from pynput import keyboard  # pip install pynput

from cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation

# Import your existing helpers
from .generate_cdpr_dataset import build_wrapper_if_needed
from .synthetic_tasks import clamp_xyz, task_language


HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "datasets" / "cdpr_synth"
VIDEO_DIR = DATASET_ROOT / "videos"
WRAP_DIR = HERE / "wrappers"


def ensure_dirs():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    WRAP_DIR.mkdir(parents=True, exist_ok=True)


def episode_out_dir(wrapper_xml: Path, task_name: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"HUMAN_CONTROL_{wrapper_xml.stem}_{task_name}_{stamp}"
    return VIDEO_DIR / base


# ---------------- Wrapper portability: patch <include file="..."> ----------------

_INCLUDE_RE = re.compile(r'(<include\s+file=")([^"]+)(")', re.IGNORECASE)

def patch_wrapper_includes(wrapper_xml: Path, local_wrap_dir: Path) -> Path:
    """
    Patch a wrapper XML so that any <include file="..."> paths that are absolute
    or point to /root/... get rewritten to local_wrap_dir/<basename>.

    Returns: path to patched wrapper XML (may be same path if no changes).
    """
    text = wrapper_xml.read_text(encoding="utf-8")

    changed = False
    def repl(match):
        nonlocal changed
        prefix, path_str, suffix = match.group(1), match.group(2), match.group(3)
        p = Path(path_str)

        # If include is already relative, keep it (MuJoCo resolves relative to XML dir)
        if not p.is_absolute():
            return match.group(0)

        # Absolute include: rewrite to local wrappers basename
        basename = p.name
        candidate = local_wrap_dir / basename
        if candidate.exists():
            changed = True
            # Use relative include for portability (relative to wrapper_xml directory)
            rel = candidate.relative_to(wrapper_xml.parent)
            return f'{prefix}{rel.as_posix()}{suffix}'

        # If basename isn't in local wrappers, still rewrite to local absolute as fallback
        changed = True
        return f'{prefix}{candidate.as_posix()}{suffix}'

    patched = _INCLUDE_RE.sub(repl, text)

    # Also common case: hardcoded "/root/repo/..." appearing in other places.
    # We *only* patch if it points into wrappers and the local file exists.
    # (This is conservative.)
    if "/root/" in patched:
        # Try to replace any /root/.../<something>.xml with local wrappers if file exists
        def replace_root_paths(m):
            nonlocal changed
            full = m.group(0)
            basename = Path(full).name
            candidate = local_wrap_dir / basename
            if candidate.exists():
                changed = True
                rel = candidate.relative_to(wrapper_xml.parent)
                return rel.as_posix()
            return full

        patched2 = re.sub(r"/root/[^\"'\s>]+\.xml", replace_root_paths, patched)
        patched = patched2

    if not changed:
        return wrapper_xml

    out = wrapper_xml.with_name(wrapper_xml.stem + "__localpatched.xml")
    out.write_text(patched, encoding="utf-8")
    print(f"🛠 Patched wrapper includes:\n  in:  {wrapper_xml}\n  out: {out}")
    return out


# ---------------- Keyboard handling ----------------

class KeyState:
    def __init__(self):
        self.chars = set()
        self.special = set()
        self.quit = False

    def on_press(self, key):
        try:
            ch = key.char
            if ch:
                self.chars.add(ch.lower())
        except AttributeError:
            self.special.add(key)
            if key == keyboard.Key.esc:
                self.quit = True

    def on_release(self, key):
        try:
            ch = key.char
            if ch and ch.lower() in self.chars:
                self.chars.remove(ch.lower())
        except AttributeError:
            if key in self.special:
                self.special.remove(key)

        # q quits
        try:
            if key == keyboard.KeyCode.from_char("q"):
                self.quit = True
        except Exception:
            pass


# ---------------- Teleop loop ----------------

def teleop_episode(sim,
                   language_instruction: str,
                   wrapper_xml: Path,
                   out_dir: Path,
                   translational_step=0.01,
                   z_step=0.01,
                   yaw_step_deg=5.0,
                   hz=30.0):
    """
    Run one teleop episode. Records every simulation step (capture_frame=True).
    """
    ks = KeyState()
    listener = keyboard.Listener(on_press=ks.on_press, on_release=ks.on_release)
    listener.start()
    
    sim.hold_current_pose(warm_steps=10)

    # Start state
    ee = sim.get_end_effector_position().copy()
    yaw = 0.0
    if hasattr(sim, "get_yaw"):
        try:
            yaw = float(sim.get_yaw())
        except Exception:
            yaw = 0.0

    yaw_step = math.radians(float(yaw_step_deg))
    dt = 1.0 / float(hz)
    last = time.time()
    step = 0

    print("🎮 Teleop controls: WASD XY, ↑/↓ Z, [/] yaw, X open, C close, Q/ESC quit")

    while not ks.quit:
        now = time.time()
        if now - last < dt:
            time.sleep(0.001)
            continue
        last = now

        dx = dy = dz = 0.0
        dyaw = 0.0

        # XY
        if "w" in ks.chars: dy += translational_step
        if "s" in ks.chars: dy -= translational_step
        if "a" in ks.chars: dx -= translational_step
        if "d" in ks.chars: dx += translational_step

        # Z
        if keyboard.Key.up in ks.special:   dz += z_step
        if keyboard.Key.down in ks.special: dz -= z_step

        # yaw
        if "[" in ks.chars: dyaw -= yaw_step
        if "]" in ks.chars: dyaw += yaw_step

        # gripper
        if "x" in ks.chars and hasattr(sim, "open_gripper"):
            sim.open_gripper()
        if "c" in ks.chars and hasattr(sim, "close_gripper"):
            sim.close_gripper()

        ee = ee + np.array([dx, dy, dz], dtype=float)
        ee = clamp_xyz(ee)

        # Apply targets
        if hasattr(sim, "set_end_effector_target"):
            sim.set_end_effector_target(ee)
        elif hasattr(sim, "set_ee_target"):
            sim.set_ee_target(ee)
        else:
            sim.set_target_position(ee)

        yaw += dyaw
        if hasattr(sim, "set_yaw"):
            try:
                sim.set_yaw(yaw)
            except Exception:
                pass

        sim.run_simulation_step(capture_frame=True)

        # --- LIVE PREVIEW (OpenCV) ---
        # use the latest captured frames (already flipped upright in capture_frame)
        if sim.overview_frames:
            ov = sim.overview_frames[-1]
            cv2.imshow("CDPR Overview", ov[:, :, ::-1])  # RGB -> BGR
        if sim.ee_camera_frames:
            ee_img = sim.ee_camera_frames[-1]
            cv2.imshow("CDPR EE Camera", ee_img[:, :, ::-1])  # RGB -> BGR

        # Allow OpenCV to process window events
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):   # quit from OpenCV window too
            ks.quit = True

        step += 1

        if step % int(hz * 2) == 0:
            print(f"step={step} ee={ee} yaw_cmd={yaw:.2f} yaw_qpos={sim.get_yaw():.2f} ctrl={sim.data.ctrl[sim.act_yaw]:.2f}")


    listener.stop()
    cv2.destroyAllWindows()

    setattr(sim, "language_instruction", language_instruction)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    sim.save_trajectory_results(str(out_dir), out_dir.name)
    print(f"✅ Saved teleop episode: {out_dir}")


# ---------------- CLI ----------------

def load_catalog(catalog_path: str):
    import yaml
    with open(catalog_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--catalog", type=str, default=None)
    ap.add_argument("--scene", type=str, default=None)
    ap.add_argument("--object", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=1)

    ap.add_argument("--task_name", type=str, default="put_into_bowl",
                    help="Used only to auto-generate language instruction if --language is not set.")
    ap.add_argument("--trans_step", type=float, default=0.01)
    ap.add_argument("--z_step", type=float, default=0.01)
    ap.add_argument("--yaw_step_deg", type=float, default=5.0)
    ap.add_argument("--hz", type=float, default=30.0)

    ap.add_argument("--force_rebuild_wrapper", action="store_true",
                    help="Ignore cached wrapper and rebuild locally.")
    ap.add_argument("--patch_wrapper_paths", action="store_true", default=True,
                    help="Patch absolute include paths inside wrapper XML to local wrappers directory.")
    ap.add_argument("--wrapper_xml", type=str, default=None,
                help="Use an existing wrapper XML and skip scene switcher.")
    ap.add_argument(
        "--language", "--instruction",
        dest="language",
        type=str,
        default=None,
        help="Natural language instruction saved into summary.txt and trajectory_data.npz (task_description)."
    )

    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    # Build scene list
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
                scene_name = str(entry)
                object_names = []
            scene_specs.append((scene_name, object_names, defaults))
    else:
        if not args.scene or not args.object:
            raise SystemExit("Provide --catalog or both --scene and --object.")
        scene_specs.append((args.scene, [args.object], {}))

    for scene_name, object_names, defaults in scene_specs:
        # Wrapper settings
        scene_z = defaults.get("scene_z", -0.85)
        ee_start = list(defaults.get("ee_start", (0.0, 0.0, 0.45)))
        table_z = defaults.get("table_z", 0.15)
        settle_t = defaults.get("settle_time", 1.0)

        # If force rebuild: delete cached wrapper(s) for this scene (simple heuristic)
        if args.force_rebuild_wrapper and not args.wrapper_xml:
            for p in WRAP_DIR.glob(f"{scene_name}__*_wrapper.xml"):
                try:
                    p.unlink()
                    print(f"🧹 Removed cached wrapper: {p}")
                except Exception as e:
                    print(f"Could not remove {p}: {e}")

        # Choose wrapper
        if args.wrapper_xml:
            wrapper_xml = Path(args.wrapper_xml).expanduser().resolve()
            if not wrapper_xml.exists():
                raise SystemExit(f"--wrapper_xml not found: {wrapper_xml}")
        else:
            wrapper_xml = build_wrapper_if_needed(
                scene_name,
                object_names,
                scene_z=scene_z,
                ee_start=ee_start,
                table_z=table_z,
                settle_time=settle_t,
            )

        # Patch wrapper paths if needed (solves /root/... includes)
        wrapper_to_use = patch_wrapper_includes(wrapper_xml, WRAP_DIR) if args.patch_wrapper_paths else wrapper_xml
        print(f"✅ Scene '{scene_name}' wrapper: {wrapper_to_use}")

        obj_name = object_names[0] if object_names else "object"
        language = args.language if args.language else task_language(args.task_name, obj_name)

        for epi in range(args.episodes):
            print(f"\n=== Teleop episode {epi+1}/{args.episodes}: scene={scene_name} obj={obj_name} ===")

            out_dir = episode_out_dir(wrapper_to_use, "teleop")
            out_dir.mkdir(parents=True, exist_ok=True)

            sim = HeadlessCDPRSimulation(xml_path=str(wrapper_to_use), output_dir=str(out_dir))
            sim.initialize()
            
            try:
                teleop_episode(
                    sim,
                    language_instruction=language,
                    wrapper_xml=wrapper_to_use,
                    out_dir=out_dir,
                    translational_step=args.trans_step,
                    z_step=args.z_step,
                    yaw_step_deg=args.yaw_step_deg,
                    hz=args.hz,
                )
            finally:
                sim.cleanup()



if __name__ == "__main__":
    main()
