#!/usr/bin/env python3
"""
Teleop data collection for CDPR + desk texture randomization/augmentation.

What it does:
- Builds/uses a cached wrapper XML (same as teleop_cdpr_collect.py).
- Creates K wrapper variants, each with a different desk/table texture.
- You teleop ONCE on the first variant. We record your command stream.
- We then replay the same command stream on the other variants to create
  K demonstrations with different tabletop appearance.

This is the common "domain randomization" trick used in many VLA / imitation setups.

Run (examples):
  python -m cdpr_dataset.teleop_cdpr_collect_randdesk \
      --scene kitchen --object milk \
      --desk_textures_dir "/Users/<you>/Desktop/.../LIBERO/libero/libero/assets/textures" \
      --desk_augments 5

or with your catalog:
  python -m cdpr_dataset.teleop_cdpr_collect_randdesk \
      --catalog path/to/catalog.yaml \
      --desk_textures_dir "/.../LIBERO/libero/libero/assets/textures" \
      --desk_augments 5
"""

import argparse
import math
import os
import random
import re
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from pynput import keyboard

from cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation

# reuse your existing helpers (same as teleop_cdpr_collect.py)
from .generate_cdpr_dataset import build_wrapper_if_needed
from .synthetic_tasks import clamp_xyz, task_language


HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "datasets" / "cdpr_synth_pick"
VIDEO_DIR = DATASET_ROOT / "videos"
WRAP_DIR = HERE / "wrappers"
DEFAULT_ALLOWED_OBJECTS = ("ycb_apple", "ycb_pear", "ycb_peach")


# ---------------- I/O ----------------

def ensure_dirs():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    WRAP_DIR.mkdir(parents=True, exist_ok=True)


def episode_out_dir(wrapper_xml: Path, task_name: str, tex_tag: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"HUMAN_CONTROL_{stamp}_{wrapper_xml.stem}_{task_name}_{tex_tag}"
    return VIDEO_DIR / base


# ---------------- Wrapper portability: patch <include file="..."> ----------------

_INCLUDE_RE = re.compile(r'(<include\s+file=")([^"]+)(")', re.IGNORECASE)

def patch_wrapper_includes(wrapper_xml: Path, local_wrap_dir: Path) -> Path:
    """
    Same idea as your existing teleop_cdpr_collect.py: rewrite absolute include paths
    to local WRAP_DIR basenames to make wrappers portable.
    """
    text = wrapper_xml.read_text(encoding="utf-8")

    changed = False

    def repl(match):
        nonlocal changed
        prefix, path_str, suffix = match.group(1), match.group(2), match.group(3)
        p = Path(path_str)

        if not p.is_absolute():
            return match.group(0)

        basename = p.name
        candidate = local_wrap_dir / basename
        if candidate.exists():
            changed = True
            rel = candidate.relative_to(wrapper_xml.parent)
            return f'{prefix}{rel.as_posix()}{suffix}'

        changed = True
        return f'{prefix}{candidate.as_posix()}{suffix}'

    patched = _INCLUDE_RE.sub(repl, text)

    if "/root/" in patched:
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

        patched = re.sub(r"/root/[^\"'\s>]+\.xml", replace_root_paths, patched)

    if not changed:
        return wrapper_xml

    out = wrapper_xml.with_name(wrapper_xml.stem + "__localpatched.xml")
    out.write_text(patched, encoding="utf-8")
    print(f"🛠 Patched wrapper includes:\n  in:  {wrapper_xml}\n  out: {out}")
    return out


# ---------------- Desk texture patching ----------------

@dataclass
class DeskPatchResult:
    wrapper_xml: Path
    matched_geoms: int


def _ensure_asset_first(root: ET.Element) -> ET.Element:
    """
    Ensure <asset> exists and is the first child of <mujoco>, so materials/textures are
    defined before any <include> that may reference them.
    """
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
        return asset

    # Move it to the front if needed
    children = list(root)
    idx = children.index(asset)
    if idx != 0:
        root.remove(asset)
        root.insert(0, asset)
    return asset


def _iter_includes(tree_root: ET.Element):
    for inc in tree_root.iter("include"):
        f = inc.get("file")
        if f:
            yield inc, f


def _resolve_include_path(current_xml: Path, file_attr: str) -> Path:
    p = Path(file_attr)
    if p.is_absolute():
        return p
    return (current_xml.parent / p).resolve()


def _relpath_or_abs(target: Path, base_dir: Path) -> str:
    try:
        return target.relative_to(base_dir).as_posix()
    except Exception:
        return target.as_posix()


def _geom_looks_like_table(geom: ET.Element) -> bool:
    """
    Optional heuristic if names/classes don't match:
    - Large box with thin z
    """
    size = geom.get("size")
    gtype = (geom.get("type") or "").lower()
    if not size:
        return False
    try:
        vals = [float(x) for x in size.replace(",", " ").split()]
        if len(vals) < 3:
            return False
        sx, sy, sz = vals[0], vals[1], vals[2]
    except Exception:
        return False

    # Typical tabletop: big in x/y, thin in z
    if gtype in ("box", "") and sx > 0.15 and sy > 0.15 and sz < 0.06:
        return True
    return False


def _patch_one_xml_file(
    orig_xml: Path,
    variant_tag: str,
    desk_mat_name: str,
    table_regex: re.Pattern,
    mapping: dict[Path, Path],
    visited: set[Path],
    force: bool,
) -> int:
    """
    Create a patched copy of orig_xml in the SAME directory:
      orig.stem__desktex_<tag>.xml

    - Rewrites its <include file="..."> tags to point to the patched copies too.
    - Sets material=desk_mat_name on matching geoms.
    Returns number of matched geoms in THIS file.
    """
    orig_xml = orig_xml.resolve()
    if orig_xml in visited:
        return 0
    visited.add(orig_xml)

    new_xml = orig_xml.with_name(f"{orig_xml.stem}__desktex_{variant_tag}{orig_xml.suffix}")
    mapping[orig_xml] = new_xml

    if new_xml.exists() and (not force):
        # We still want to count matches? Keep it simple: assume already patched.
        return 0

    try:
        tree = ET.parse(orig_xml)
    except Exception as e:
        print(f"⚠️ Could not parse XML (skipping): {orig_xml}\n   {e}")
        return 0

    root = tree.getroot()

    # 1) recurse into includes first, so we can rewrite include paths
    for inc_elem, f in list(_iter_includes(root)):
        inc_path = _resolve_include_path(orig_xml, f)
        if inc_path.exists():
            _patch_one_xml_file(
                inc_path, variant_tag, desk_mat_name, table_regex, mapping, visited, force
            )
            patched_child = mapping.get(inc_path, inc_path)
            inc_elem.set("file", _relpath_or_abs(patched_child, orig_xml.parent))
        else:
            # leave it alone, but warn
            print(f"⚠️ include not found from {orig_xml.name}: {f}")

    # 2) patch table/desk geoms
    matched = 0
    for geom in root.iter("geom"):
        name = (geom.get("name") or "")
        cls  = (geom.get("class") or "")
        mat  = (geom.get("material") or "")

        if table_regex.search(name) or table_regex.search(cls) or table_regex.search(mat) or _geom_looks_like_table(geom):
            geom.set("material", desk_mat_name)
            matched += 1

    # Write patched file
    try:
        tree.write(new_xml, encoding="utf-8", xml_declaration=True)
    except Exception as e:
        print(f"⚠️ Could not write patched XML: {new_xml}\n   {e}")
        return matched

    return matched

import numpy as np
import cv2
from pathlib import Path

def make_tiled_texture(
    src: Path,
    dst: Path,
    tiles_x: int,
    tiles_y: int,
    max_tex_size: int = 8192,          # try 8192 or 16384
    min_px_per_tile: int = 256,        # increase to 512 for extra sharpness
):
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read texture: {src}")

    h, w = img.shape[:2]

    # 1) If the tile is *too big* to fit repeats under max_tex_size, downscale the tile (sharply).
    #    Compute the largest tile size that fits.
    max_tile_w = max(1, max_tex_size // tiles_x)
    max_tile_h = max(1, max_tex_size // tiles_y)

    # Keep aspect ratio
    scale = min(1.0, max_tile_w / w, max_tile_h / h)

    # 2) Also ensure each tile has at least min_px_per_tile (if possible under max_tex_size).
    #    If min_px_per_tile would exceed max_tex_size, it will be capped by max_tile_w/h above.
    desired_scale = max(min_px_per_tile / w, min_px_per_tile / h)
    scale = min(1.0, max(scale, min(desired_scale, 1.0)))

    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        # Sharper than INTER_AREA for “keep it crisp”
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = img.shape[:2]

    # 3) Tile (this step is lossless)
    if img.ndim == 3:
        tiled = np.tile(img, (tiles_y, tiles_x, 1))
    else:
        tiled = np.tile(img, (tiles_y, tiles_x))

    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), tiled)
    if not ok:
        raise RuntimeError(f"Could not write tiled texture: {dst}")
    
def make_textured_wrapper(
    base_wrapper_xml: Path,
    texture_png: Path,
    variant_tag: str,
    table_geom_regex: str,
    texrepeat_xy: tuple[int, int],
    force: bool = False,
) -> DeskPatchResult:
    """
    Produces a wrapper variant with:
      - <asset> containing a unique texture+material referencing texture_png (copied locally)
      - patched copies of included XMLs with table geoms forced to that material
    """
    base_wrapper_xml = base_wrapper_xml.resolve()
    texture_png = texture_png.resolve()

    # Copy texture into wrappers/_desk_textures for portability/repro
    tex_dir = WRAP_DIR / "_desk_textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    copied_tex = tex_dir / f"{variant_tag}__{texture_png.name}"
    # if (not copied_tex.exists()) or force:
        # shutil.copy2(texture_png, copied_tex)
    copied_tex = tex_dir / f"{variant_tag}__tiled_{texrepeat_xy[0]}x{texrepeat_xy[1]}__{texture_png.stem}.png"
    if (not copied_tex.exists()) or force:
        make_tiled_texture(texture_png, copied_tex, texrepeat_xy[0], texrepeat_xy[1])

    # Names must be unique per variant to avoid collisions
    desk_tex_name = f"desktex_{variant_tag}"
    desk_mat_name = f"deskmat_{variant_tag}"

    # Patch wrapper itself into a copy in same directory
    mapping: dict[Path, Path] = {}
    visited: set[Path] = set()
    table_regex = re.compile(table_geom_regex, re.IGNORECASE)

    # First: make patched copies of wrapper + includes with geom material assignment
    matched_total = _patch_one_xml_file(
        base_wrapper_xml,
        variant_tag,
        desk_mat_name,
        table_regex,
        mapping,
        visited,
        force=force,
    )
    wrapper_copy = mapping.get(base_wrapper_xml, base_wrapper_xml)
    if not wrapper_copy.exists():
        raise RuntimeError(f"Failed to create wrapper copy: {wrapper_copy}")

    # Second: inject the asset (texture/material) into wrapper_copy
    tree = ET.parse(wrapper_copy)
    root = tree.getroot()
    asset = _ensure_asset_first(root)

    # Add <texture> and <material>
    # Use relative path from wrapper to copied texture
    tex_file_attr = _relpath_or_abs(copied_tex, wrapper_copy.parent)

    # --- texture ---
    tex_el = None
    for el in asset.findall("texture"):
        if el.get("name") == desk_tex_name:
            tex_el = el
            break
    if tex_el is None:
        tex_el = ET.SubElement(asset, "texture", {"name": desk_tex_name, "type": "2d"})
    tex_el.set("file", tex_file_attr)

    # --- material ---
    mat_el = None
    for el in asset.findall("material"):
        if el.get("name") == desk_mat_name:
            mat_el = el
            break
    if mat_el is None:
        mat_el = ET.SubElement(asset, "material", {"name": desk_mat_name})

    mat_el.set("texture", desk_tex_name)
    # mat_el.set("texrepeat", f"{texrepeat_xy[0]} {texrepeat_xy[1]}")
    # mat_el.set("texuniform", "false")   # or "true" if you actually want repeats-per-meter
    mat_el.set("texrepeat", "1 1")
    mat_el.set("texuniform", "false")
    # If re-running, avoid duplicating tags
    def _has_asset(tag, name):
        for el in asset.findall(tag):
            if el.get("name") == name:
                return True
        return False

    if not _has_asset("texture", desk_tex_name):
        ET.SubElement(asset, "texture", {
            "name": desk_tex_name,
            "type": "2d",
            "file": tex_file_attr
        })

    if not _has_asset("material", desk_mat_name):
        ET.SubElement(asset, "material", {
            "name": desk_mat_name,
            "texture": desk_tex_name,
            "texrepeat": f"{texrepeat_xy[0]} {texrepeat_xy[1]}",
            "texuniform": "false",
        })

    tree.write(wrapper_copy, encoding="utf-8", xml_declaration=True)

    if matched_total == 0:
        print("⚠️ WARNING: Did not match any desk/table geoms by regex/heuristic.")
        print(f"   regex={table_geom_regex}")
        print("   You likely need to adjust --desk_geom_regex to match your tabletop geom/material names.")

    return DeskPatchResult(wrapper_xml=wrapper_copy, matched_geoms=matched_total)


# ---------------- Keyboard handling + Teleop + Replay ----------------

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

        try:
            if key == keyboard.KeyCode.from_char("q"):
                self.quit = True
        except Exception:
            pass


def _apply_targets(sim, ee_xyz, yaw, gripper_open: bool | None):
    if hasattr(sim, "set_end_effector_target"):
        sim.set_end_effector_target(ee_xyz)
    elif hasattr(sim, "set_ee_target"):
        sim.set_ee_target(ee_xyz)
    else:
        sim.set_target_position(ee_xyz)

    if hasattr(sim, "set_yaw"):
        try:
            sim.set_yaw(float(yaw))
        except Exception:
            pass

    if gripper_open is True and hasattr(sim, "open_gripper"):
        sim.open_gripper()
    elif gripper_open is False and hasattr(sim, "close_gripper"):
        sim.close_gripper()


def teleop_episode_record(
    sim,
    language_instruction: str,
    out_dir: Path,
    translational_step=0.01,
    z_step=0.01,
    yaw_step_deg=5.0,
    hz=30.0,
    show_preview=True,
):
    """
    Teleop once; record per-step commands so we can replay them with different desk textures.
    Returns: list of dicts with keys: ee(3,), yaw(float), gripper_open(bool|None)
    """
    ks = KeyState()
    listener = keyboard.Listener(on_press=ks.on_press, on_release=ks.on_release)
    listener.start()

    sim.hold_current_pose(warm_steps=10)

    ee = sim.get_end_effector_position().copy()
    yaw = 0.0
    try:
        if hasattr(sim, "get_yaw"):
            yaw = float(sim.get_yaw())
    except Exception:
        yaw = 0.0

    yaw_step = math.radians(float(yaw_step_deg))
    dt = 1.0 / float(hz)
    last = time.time()
    step = 0

    gripper_open: bool | None = None
    commands = []

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

        # gripper (stateful)
        if "x" in ks.chars:
            gripper_open = True
        if "c" in ks.chars:
            gripper_open = False

        ee = ee + np.array([dx, dy, dz], dtype=float)
        ee = clamp_xyz(ee)
        yaw += dyaw

        _apply_targets(sim, ee, yaw, gripper_open)

        sim.run_simulation_step(capture_frame=True)

        # record AFTER stepping so it's aligned with what was commanded this step
        commands.append({
            "ee": ee.copy(),
            "yaw": float(yaw),
            "gripper_open": gripper_open,
        })

        if show_preview:
            if sim.overview_frames:
                ov = sim.overview_frames[-1]
                cv2.imshow("CDPR Overview", ov[:, :, ::-1])
            if sim.ee_camera_frames:
                ee_img = sim.ee_camera_frames[-1]
                cv2.imshow("CDPR EE Camera", ee_img[:, :, ::-1])

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                ks.quit = True

        step += 1
        if step % int(hz * 2) == 0 and hasattr(sim, "get_yaw"):
            try:
                print(f"step={step} ee={ee} yaw={yaw:.2f} yaw_qpos={sim.get_yaw():.2f}")
            except Exception:
                pass

    listener.stop()
    cv2.destroyAllWindows()

    setattr(sim, "language_instruction", language_instruction)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim.save_trajectory_results(str(out_dir), out_dir.name)
    print(f"✅ Saved teleop episode: {out_dir}")

    return commands


def replay_episode(
    sim,
    commands,
    language_instruction: str,
    out_dir: Path,
    show_preview=False,
):
    """
    Replay a recorded command stream to generate an augmented demo with different visuals.
    """
    sim.hold_current_pose(warm_steps=10)

    for cmd in commands:
        _apply_targets(sim, cmd["ee"], cmd["yaw"], cmd.get("gripper_open", None))
        sim.run_simulation_step(capture_frame=True)

        if show_preview:
            if sim.overview_frames:
                ov = sim.overview_frames[-1]
                cv2.imshow("CDPR Overview (replay)", ov[:, :, ::-1])
            if sim.ee_camera_frames:
                ee_img = sim.ee_camera_frames[-1]
                cv2.imshow("CDPR EE Camera (replay)", ee_img[:, :, ::-1])
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    if show_preview:
        cv2.destroyAllWindows()

    setattr(sim, "language_instruction", language_instruction)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim.save_trajectory_results(str(out_dir), out_dir.name)
    print(f"✅ Saved replay episode: {out_dir}")


# ---------------- CLI ----------------

def _safe_unlink(path: Path):
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        pass


def _cleanup_generated_wrappers(created_paths: list[Path], variant_tags: list[str]):
    to_remove = {p.resolve() for p in created_paths if p is not None}
    tex_dir = WRAP_DIR / "_desk_textures"

    for tag in variant_tags:
        for xml_path in WRAP_DIR.rglob(f"*__desktex_{tag}.xml"):
            to_remove.add(xml_path.resolve())
        if tex_dir.exists():
            for tex_path in tex_dir.glob(f"{tag}__*"):
                to_remove.add(tex_path.resolve())

    for p in sorted(to_remove):
        _safe_unlink(p)


def load_catalog(catalog_path: str):
    import yaml
    with open(catalog_path, "r") as f:
        return yaml.safe_load(f)


def _guess_textures_dir() -> Path | None:
    # You can hardcode your own guess here if you want; keep it conservative.
    env = (
        os.environ.get("LIBERO_TEXTURES_DIR")
        or os.environ.get("LIBERO_ASSETS_TEXTURES_DIR")
    )
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    return None


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--catalog", type=str, default=None)
    ap.add_argument("--scene", type=str, default=None)
    ap.add_argument("--object", type=str, default=None)
    ap.add_argument(
        "--allowed_objects",
        type=str,
        nargs="*",
        default=list(DEFAULT_ALLOWED_OBJECTS),
        help="Only use/track these object names from the catalog.",
    )

    ap.add_argument("--episodes", type=int, default=1,
                    help="How many base teleop recordings to do (each base recording gets desk_augments variants).")

    ap.add_argument("--task_name", type=str, default="put_into_bowl",
                    help="Only used to auto-generate language instruction if --language is not set.")
    ap.add_argument("--trans_step", type=float, default=0.01)
    ap.add_argument("--z_step", type=float, default=0.01)
    ap.add_argument("--yaw_step_deg", type=float, default=5.0)
    ap.add_argument("--hz", type=float, default=30.0)

    ap.add_argument("--force_rebuild_wrapper", action="store_true")
    ap.add_argument("--patch_wrapper_paths", action="store_true", default=True)
    ap.add_argument("--wrapper_xml", type=str, default=None)

    ap.add_argument("--language", "--instruction", dest="language", type=str, default=None)

    # desk randomization
    ap.add_argument("--desk_textures_dir", type=str, required=True,
                    help="Path to LIBERO/libero/libero/assets/textures (or your own texture folder).")
    ap.add_argument("--desk_augments", type=int, default=5,
                    help="Total number of desk-texture variants per teleop recording (including the teleop one).")
    ap.add_argument("--desk_geom_regex", type=str, default=r"(table|desk|workbench|counter|surface)",
                    help="Regex used to decide which geoms/materials are 'the desk/table'.")
    ap.add_argument("--desk_texrepeat", type=int, nargs=2, default=(20, 20),
                    help="Material texrepeat X Y for the desk texture tiling.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for picking textures (0 means use time-based seed).")
    ap.add_argument("--force_regen_textured_wrappers", action="store_true",
                    help="Re-generate textured wrapper XML copies even if they already exist.")
    ap.add_argument(
        "--keep_wrappers",
        action="store_true",
        help="Keep generated temporary wrappers/textured copies instead of deleting them.",
    )
    ap.add_argument(
                    "--desk_textures",
                    type=str,
                    nargs="*",
                    default=None,
                    help="Optional explicit list of texture filenames (or absolute paths). If set, overrides random sampling from --desk_textures_dir."
                )
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    # RNG
    if args.seed == 0:
        random.seed()
    else:
        random.seed(int(args.seed))

    textures_dir = Path(args.desk_textures_dir).expanduser().resolve()
    if not textures_dir.exists():
        raise SystemExit(f"--desk_textures_dir not found: {textures_dir}")

    texture_files = sorted([p for p in textures_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not texture_files:
        raise SystemExit(f"No textures found in: {textures_dir}")

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

    allowed_objects = {str(x) for x in args.allowed_objects}
    if not allowed_objects:
        raise SystemExit("--allowed_objects cannot be empty.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    for scene_name, object_names, defaults in scene_specs:
        object_names = [str(x) for x in object_names if str(x) in allowed_objects]
        if not object_names:
            print(f"⏭️ Skipping scene '{scene_name}': none of its objects are in --allowed_objects.")
            continue

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

        obj_name = object_names[0] if object_names else "object"
        language = args.language if args.language else task_language(args.task_name, obj_name)

        for epi in range(args.episodes):
            print(f"\n=== Teleop recording {epi+1}/{args.episodes}: scene={scene_name} obj={obj_name} ===")
            created_paths: list[Path] = []
            variant_tags: list[str] = []

            try:
                if args.wrapper_xml:
                    src_wrapper_xml = Path(args.wrapper_xml).expanduser().resolve()
                    if not src_wrapper_xml.exists():
                        raise SystemExit(f"--wrapper_xml not found: {src_wrapper_xml}")
                    wrapper_xml = WRAP_DIR / f"{src_wrapper_xml.stem}__teleopsrc_{int(time.time_ns())}.xml"
                    shutil.copy2(src_wrapper_xml, wrapper_xml)
                    created_paths.append(wrapper_xml)
                else:
                    wrapper_out = WRAP_DIR / (
                        f"{scene_name}__{'-'.join(sorted(object_names))}__teleoptmp_{int(time.time_ns())}.xml"
                    )
                    wrapper_xml = build_wrapper_if_needed(
                        scene_name,
                        object_names,
                        scene_z=scene_z,
                        ee_start=ee_start,
                        table_z=table_z,
                        settle_time=settle_t,
                        wrapper_out=wrapper_out,
                        use_cache=False,
                    )
                    created_paths.append(wrapper_xml)

                wrapper_base = patch_wrapper_includes(wrapper_xml, WRAP_DIR) if args.patch_wrapper_paths else wrapper_xml
                if wrapper_base != wrapper_xml:
                    created_paths.append(wrapper_base)
                print(f"✅ Scene '{scene_name}' wrapper: {wrapper_base}")

                # choose textures
                k = int(args.desk_augments)
                if k <= 0:
                    raise SystemExit("--desk_augments must be >= 1")

                if args.desk_textures and len(args.desk_textures) > 0:
                    chosen = []
                    for t in args.desk_textures:
                        p = Path(t).expanduser()
                        if not p.is_absolute():
                            p = (textures_dir / p).resolve()
                        if not p.exists():
                            raise SystemExit(f"Texture not found: {p}")
                        chosen.append(p)
                    if len(chosen) != k:
                        raise SystemExit(f"--desk_textures count ({len(chosen)}) must equal --desk_augments ({k})")
                else:
                    if k <= len(texture_files):
                        chosen = random.sample(texture_files, k)
                    else:
                        chosen = [random.choice(texture_files) for _ in range(k)]

                # build textured wrappers
                wrapper_variants = []
                for vi, tex in enumerate(chosen):
                    tag = f"{run_id}_epi{epi+1}_v{vi+1}_{tex.stem}"
                    variant_tags.append(tag)
                    res = make_textured_wrapper(
                        wrapper_base,
                        tex,
                        variant_tag=tag,
                        table_geom_regex=args.desk_geom_regex,
                        texrepeat_xy=tuple(args.desk_texrepeat),
                        force=args.force_regen_textured_wrappers,
                    )
                    wrapper_variants.append((res.wrapper_xml, tex, tag, res.matched_geoms))
                    created_paths.append(res.wrapper_xml)
                    print(
                        f"  🎨 variant {vi+1}/{k}: {tex.name} -> {res.wrapper_xml.name} "
                        f"(matched_geoms={res.matched_geoms})"
                    )

                # --- 1) Teleop ONCE on the first variant ---
                wrapper0, tex0, tag0, _ = wrapper_variants[0]
                out0 = episode_out_dir(wrapper0, "teleop", tex_tag=tex0.stem)
                out0.mkdir(parents=True, exist_ok=True)
                (out0 / "desk_texture.txt").write_text(str(tex0), encoding="utf-8")

                sim0 = HeadlessCDPRSimulation(xml_path=str(wrapper0), output_dir=str(out0))
                sim0.initialize()
                try:
                    commands = teleop_episode_record(
                        sim0,
                        language_instruction=language,
                        out_dir=out0,
                        translational_step=args.trans_step,
                        z_step=args.z_step,
                        yaw_step_deg=args.yaw_step_deg,
                        hz=args.hz,
                        show_preview=True,
                    )
                finally:
                    sim0.cleanup()

                # --- 2) Replay the same commands on remaining variants ---
                for wrapper_i, tex_i, tag_i, _ in wrapper_variants[1:]:
                    out_i = episode_out_dir(wrapper_i, "replay", tex_tag=tex_i.stem)
                    out_i.mkdir(parents=True, exist_ok=True)
                    (out_i / "desk_texture.txt").write_text(str(tex_i), encoding="utf-8")

                    sim_i = HeadlessCDPRSimulation(xml_path=str(wrapper_i), output_dir=str(out_i))
                    sim_i.initialize()
                    try:
                        replay_episode(
                            sim_i,
                            commands=commands,
                            language_instruction=language,
                            out_dir=out_i,
                            show_preview=False,
                        )
                    finally:
                        sim_i.cleanup()
            finally:
                if not args.keep_wrappers:
                    _cleanup_generated_wrappers(created_paths, variant_tags)


if __name__ == "__main__":
    main()
