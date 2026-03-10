from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import hashlib
import re
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml

try:
    import mujoco as mj
except Exception:  # pragma: no cover - optional runtime dependency
    mj = None

try:
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - optional runtime dependency
    gym = None
    spaces = None

from .rl_instruction_tasks import (
    INSTRUCTION_TYPES,
    compute_instruction_reward,
    init_reward_state,
    instruction_to_onehot,
    sample_instruction,
)
from .synthetic_tasks import (
    body_bottom_offset,
    clamp_xyz,
    infer_workspace_surface_z,
    place_objects_non_overlapping,
    resolve_body_name,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG_PATH = HERE / "datasets" / "cdpr_scene_catalog.yaml"
DEFAULT_VIDEO_DIR = HERE / "datasets" / "cdpr_synth" / "videos"
DEFAULT_ALLOWED_OBJECTS: tuple[str, ...] = ("ycb_apple", "ycb_pear", "ycb_peach")
DEFAULT_DESK_GEOM_REGEX = r"(table|desk|workbench|counter|surface)"
WRAP_DIR = HERE / "wrappers"
MIN_EE_START_Z = 0.35

ROBOT_BODY_PREFIXES = (
    "world",
    "rotor_",
    "slider_",
    "ee_",
    "camera_",
    "yaw_frame",
    "ee_platform",
    "finger_",
)


@dataclass(frozen=True)
class SceneSpec:
    name: str
    objects: tuple[str, ...]

@dataclass
class DeskTexturePatchResult:
    wrapper_xml: Path
    generated_xmls: list[Path]
    generated_files: list[Path]
    chosen_texture: Path
    matched_geoms: int


def _load_catalog(catalog_path: Path) -> tuple[dict[str, Any], list[SceneSpec]]:
    with catalog_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    defaults = dict(cfg.get("defaults", {}))
    scenes_raw = cfg.get("scenes", [])

    scenes: list[SceneSpec] = []
    for row in scenes_raw:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        objects = tuple(str(x) for x in (row.get("objects") or []))
        if name and objects:
            scenes.append(SceneSpec(name=name, objects=objects))

    if not scenes:
        raise ValueError(f"No scenes with objects found in catalog: {catalog_path}")

    return defaults, scenes


def _filter_scenes_to_allowed_objects(
    scenes: Sequence[SceneSpec], allowed_objects: Sequence[str]
) -> list[SceneSpec]:
    allowed_set = {str(x) for x in allowed_objects}
    if not allowed_set:
        return list(scenes)

    filtered: list[SceneSpec] = []
    for scene in scenes:
        objects = tuple(obj for obj in scene.objects if obj in allowed_set)
        if objects:
            filtered.append(SceneSpec(name=scene.name, objects=objects))

    if not filtered:
        raise ValueError(
            "No catalog scenes remain after allowed-object filtering. "
            f"Allowed objects: {sorted(allowed_set)}"
        )
    return filtered


def _iter_includes(tree_root: ET.Element):
    for inc in tree_root.iter("include"):
        file_attr = inc.get("file")
        if file_attr:
            yield inc, file_attr


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


def _ensure_asset_first(root: ET.Element) -> ET.Element:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
        return asset

    children = list(root)
    idx = children.index(asset)
    if idx != 0:
        root.remove(asset)
        root.insert(0, asset)
    return asset


def _geom_looks_like_table(geom: ET.Element) -> bool:
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

    return gtype in ("box", "") and sx > 0.15 and sy > 0.15 and sz < 0.06


def _patch_xml_tree_for_desk_material(
    source_xml: Path,
    variant_tag: str,
    desk_mat_name: str,
    table_regex: re.Pattern[str],
    mapping: dict[Path, Path],
    generated_xmls: list[Path],
) -> int:
    source_xml = source_xml.resolve()
    if source_xml in mapping:
        return 0

    path_hash = hashlib.sha1(source_xml.as_posix().encode("utf-8")).hexdigest()[:10]
    patched_xml = WRAP_DIR / f"{source_xml.stem}__{path_hash}__desktex_{variant_tag}{source_xml.suffix}"
    mapping[source_xml] = patched_xml

    tree = ET.parse(source_xml)
    root = tree.getroot()
    matched = 0

    for inc_elem, file_attr in list(_iter_includes(root)):
        include_src = _resolve_include_path(source_xml, file_attr)
        if not include_src.exists():
            continue
        matched += _patch_xml_tree_for_desk_material(
            include_src,
            variant_tag=variant_tag,
            desk_mat_name=desk_mat_name,
            table_regex=table_regex,
            mapping=mapping,
            generated_xmls=generated_xmls,
        )
        include_dst = mapping.get(include_src.resolve(), include_src)
        inc_elem.set("file", _relpath_or_abs(include_dst, patched_xml.parent))

    for geom in root.iter("geom"):
        name = (geom.get("name") or "")
        cls = (geom.get("class") or "")
        mat = (geom.get("material") or "")
        if table_regex.search(name) or table_regex.search(cls) or table_regex.search(mat) or _geom_looks_like_table(geom):
            geom.set("material", desk_mat_name)
            matched += 1

    tree.write(patched_xml, encoding="utf-8", xml_declaration=True)
    generated_xmls.append(patched_xml)
    return matched


def _build_textured_wrapper_variant(
    base_wrapper_xml: Path,
    chosen_texture: Path,
    variant_tag: str,
    desk_geom_regex: str,
    desk_texrepeat: tuple[int, int],
) -> DeskTexturePatchResult:
    base_wrapper_xml = base_wrapper_xml.resolve()
    chosen_texture = chosen_texture.resolve()

    tex_dir = WRAP_DIR / "_desk_textures"
    tex_dir.mkdir(parents=True, exist_ok=True)

    copied_texture = tex_dir / f"{variant_tag}__{chosen_texture.name}"
    shutil.copy2(chosen_texture, copied_texture)

    desk_tex_name = f"desktex_{variant_tag}"
    desk_mat_name = f"deskmat_{variant_tag}"
    table_regex = re.compile(desk_geom_regex, re.IGNORECASE)

    mapping: dict[Path, Path] = {}
    generated_xmls: list[Path] = []
    matched_geoms = _patch_xml_tree_for_desk_material(
        base_wrapper_xml,
        variant_tag=variant_tag,
        desk_mat_name=desk_mat_name,
        table_regex=table_regex,
        mapping=mapping,
        generated_xmls=generated_xmls,
    )

    wrapper_copy = mapping.get(base_wrapper_xml.resolve(), base_wrapper_xml.resolve())
    tree = ET.parse(wrapper_copy)
    root = tree.getroot()
    asset = _ensure_asset_first(root)

    tex_file_attr = _relpath_or_abs(copied_texture, wrapper_copy.parent)

    tex_el = None
    for el in asset.findall("texture"):
        if el.get("name") == desk_tex_name:
            tex_el = el
            break
    if tex_el is None:
        tex_el = ET.SubElement(asset, "texture", {"name": desk_tex_name, "type": "2d"})
    tex_el.set("file", tex_file_attr)

    mat_el = None
    for el in asset.findall("material"):
        if el.get("name") == desk_mat_name:
            mat_el = el
            break
    if mat_el is None:
        mat_el = ET.SubElement(asset, "material", {"name": desk_mat_name})
    mat_el.set("texture", desk_tex_name)
    mat_el.set("texrepeat", f"{int(desk_texrepeat[0])} {int(desk_texrepeat[1])}")
    mat_el.set("texuniform", "false")

    tree.write(wrapper_copy, encoding="utf-8", xml_declaration=True)

    return DeskTexturePatchResult(
        wrapper_xml=wrapper_copy,
        generated_xmls=generated_xmls,
        generated_files=[copied_texture],
        chosen_texture=chosen_texture,
        matched_geoms=matched_geoms,
    )


def _import_wrapper_builder():
    # Lazy import to avoid importing cdpr_mujoco at module import time.
    from .generate_cdpr_dataset import build_wrapper_if_needed

    return build_wrapper_if_needed


class _EnvBase:
    pass


if gym is not None:
    _EnvBase = gym.Env


class CDPRLanguageRLEnv(_EnvBase):
    """
    Language-conditioned RL environment over the CDPR MuJoCo simulation.

    Action space:
      - Box(5): [dx, dy, dz, dyaw, gripper_cmd], each in [-1, 1]
      - dx/dy/dz are delta end-effector commands scaled by action_step_xyz.
      - dyaw is scaled by action_step_yaw.
      - gripper_cmd > +0.2 closes gripper; < -0.2 opens gripper.

    Observation space:
      - ee_position: (3,)
      - target_object_position: (3,)
      - all_object_positions: (max_objects, 3)
      - object_position_mask: (max_objects,)
      - instruction_onehot: (5,)
      - goal_direction: (2,) for move instructions, zeros for pick.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        catalog_path: Path | str | None = None,
        max_steps: int = 150,
        max_objects: int = 8,
        action_step_xyz: float = 0.02,
        action_step_yaw: float = 0.25,
        move_distance: float = 0.20,
        lift_distance: float = 0.10,
        capture_frames: bool = False,
        instruction_types: Optional[Sequence[str]] = None,
        allowed_objects: Optional[Sequence[str]] = DEFAULT_ALLOWED_OBJECTS,
        desk_textures_dir: Path | str | None = None,
        desk_geom_regex: str = DEFAULT_DESK_GEOM_REGEX,
        desk_texrepeat: Sequence[int] = (20, 20),
        wrapper_cleanup: bool = True,
        use_wrapper_cache: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if gym is None or spaces is None:
            raise ImportError(
                "CDPRLanguageRLEnv requires gym (tested with gym==0.26.2). "
                "Install it before creating this env."
            )
        if mj is None:
            raise ImportError("CDPRLanguageRLEnv requires mujoco. Install it before creating this env.")

        try:
            from cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation
        except Exception as exc:  # pragma: no cover - runtime dependency check
            raise ImportError(
                "CDPRLanguageRLEnv requires cdpr_mujoco. Install it before creating this env."
            ) from exc

        self._sim_cls = HeadlessCDPRSimulation
        self.sim = None

        self.catalog_path = Path(catalog_path) if catalog_path is not None else DEFAULT_CATALOG_PATH
        self.defaults, scenes = _load_catalog(self.catalog_path)
        self.allowed_objects = tuple(str(x) for x in (allowed_objects or ()))
        self.scenes = _filter_scenes_to_allowed_objects(scenes, self.allowed_objects)

        self.max_steps = int(max_steps)
        self.max_objects = int(max_objects)
        self.action_step_xyz = float(action_step_xyz)
        self.action_step_yaw = float(action_step_yaw)
        self.move_distance = float(move_distance)
        self.lift_distance = float(lift_distance)
        self.capture_frames = bool(capture_frames)
        self.instruction_types = tuple(instruction_types) if instruction_types else None
        self.wrapper_cleanup = bool(wrapper_cleanup)
        self.use_wrapper_cache = bool(use_wrapper_cache)
        self.desk_geom_regex = str(desk_geom_regex)
        texrepeat_vals = tuple(desk_texrepeat)
        if len(texrepeat_vals) != 2:
            raise ValueError("desk_texrepeat must contain exactly two integers: X Y.")
        self.desk_texrepeat = (int(texrepeat_vals[0]), int(texrepeat_vals[1]))

        self.np_random = np.random.default_rng(seed)
        WRAP_DIR.mkdir(parents=True, exist_ok=True)
        self.wrapper_dir = WRAP_DIR

        self.desk_texture_files: list[Path] = []
        if desk_textures_dir is not None:
            tex_dir = Path(desk_textures_dir).expanduser().resolve()
            if not tex_dir.exists():
                raise ValueError(f"desk_textures_dir not found: {tex_dir}")
            self.desk_texture_files = sorted(
                p for p in tex_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            )
            if not self.desk_texture_files:
                raise ValueError(f"No texture files found in: {tex_dir}")

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "ee_position": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float32),
                "target_object_position": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float32),
                "all_object_positions": spaces.Box(
                    -2.0, 2.0, shape=(self.max_objects, 3), dtype=np.float32
                ),
                "object_position_mask": spaces.Box(
                    0.0, 1.0, shape=(self.max_objects,), dtype=np.float32
                ),
                "instruction_onehot": spaces.Box(
                    0.0, 1.0, shape=(len(INSTRUCTION_TYPES),), dtype=np.float32
                ),
                "goal_direction": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            }
        )

        self._step_count = 0
        self._yaw = 0.0
        self._last_gripper_cmd = 0.0
        self._instruction_spec = None
        self._reward_state = None
        self._scene_name = ""
        self._target_catalog_name = ""
        self._target_body_name = ""
        self._catalog_to_body: dict[str, str] = {}
        self._object_body_names: list[str] = []
        self._scene_catalog_objects: list[str] = []
        self._cleanup_paths: list[Path] = []
        self._cleanup_path_set: set[Path] = set()
        self._desk_texture_name = ""
        self._current_wrapper_xml: Path | None = None
        self._inverse_catalog_to_body: dict[str, str] = {}
        self._prev_object_positions: dict[str, np.ndarray] = {}
        self._prev_ee_for_catch = np.zeros((3,), dtype=np.float32)
        self._last_caught_body = ""
        self._last_caught_catalog = ""
        self._support_surface_z = 0.0
        self._ee_min_z = float("-inf")
        self._ee_spawn_z = float("-inf")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.close()
        scene = self._sample_scene(options=options)
        self._scene_name = scene.name
        self._scene_catalog_objects = list(scene.objects)

        wrapper_xml = self._build_wrapper(scene=scene)
        self._current_wrapper_xml = wrapper_xml
        self.sim = self._sim_cls(xml_path=str(wrapper_xml), output_dir=str(DEFAULT_VIDEO_DIR))
        self.sim.initialize()
        if hasattr(self.sim, "hold_current_pose"):
            self.sim.hold_current_pose(warm_steps=10)
        self._refresh_workspace_safety()
        self._move_ee_to_spawn_height()
        self._clear_sim_recording_buffers()

        self._catalog_to_body, self._object_body_names = self._resolve_objects(scene.objects)
        self._inverse_catalog_to_body = {v: k for k, v in self._catalog_to_body.items()}
        if self._object_body_names:
            try:
                place_objects_non_overlapping(
                    self.sim,
                    self._object_body_names,
                    xy_bounds=((-0.20, 0.20), (-0.20, 0.20), self._support_surface_z),
                    min_gap=0.02,
                    min_ee_dist=0.10,
                )
            except Exception:
                # Continue if placement fails; wrapper-provided placement is still valid.
                pass

        requested_target = (options or {}).get("target_object")
        if requested_target and requested_target in scene.objects:
            self._target_catalog_name = str(requested_target)
        else:
            idx = int(self.np_random.integers(0, len(scene.objects)))
            self._target_catalog_name = scene.objects[idx]

        target_body = self._catalog_to_body.get(self._target_catalog_name)
        if target_body is None:
            try:
                target_body = resolve_body_name(self.sim, self._target_catalog_name)
            except Exception:
                target_body = self._object_body_names[0] if self._object_body_names else ""
        self._target_body_name = target_body
        if not self._target_body_name:
            raise RuntimeError("Could not resolve target object body for RL episode reset.")

        self._instruction_spec = sample_instruction(
            target_object=self._target_catalog_name,
            rng=self.np_random,
            allowed_instruction_types=self.instruction_types,
            move_distance=self.move_distance,
            lift_distance=self.lift_distance,
        )
        setattr(self.sim, "language_instruction", self._instruction_spec.text)

        ee0 = self._get_ee_position()
        obj0 = self._get_body_position(self._target_body_name)
        self._reward_state = init_reward_state(ee0, obj0)
        self._step_count = 0
        self._yaw = self._read_current_yaw()
        self._last_gripper_cmd = 0.0
        self._prev_ee_for_catch = ee0.copy()
        self._prev_object_positions = {}
        for body_name in self._object_body_names:
            try:
                self._prev_object_positions[body_name] = self._get_body_position(body_name)
            except Exception:
                continue
        self._last_caught_body = ""
        self._last_caught_catalog = ""

        obs = self._get_obs()
        info = self._base_info()
        info["success"] = False
        return obs, info

    def step(self, action: np.ndarray):
        if self.sim is None:
            raise RuntimeError("Environment was not reset before step().")
        if self._instruction_spec is None or self._reward_state is None:
            raise RuntimeError("Internal state is missing. Call reset().")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 5:
            raise ValueError(f"Expected action shape (5,), got {action.shape}")

        action = np.clip(action, -1.0, 1.0)
        self._apply_action(action)

        ee = self._get_ee_position()
        obj = self._get_body_position(self._target_body_name)
        gripper_surface_alignment = self._get_gripper_surface_alignment(obj_pos=obj)
        camera_alignment = self._get_ee_camera_alignment(obj_pos=obj)
        ee_height_above_surface: Optional[float]
        if np.isfinite(self._support_surface_z):
            ee_height_above_surface = float(ee[2] - self._support_surface_z)
        else:
            ee_height_above_surface = None
        ee_yaw_for_reward: Optional[float] = None
        if hasattr(self.sim, "set_yaw") or hasattr(self.sim, "get_yaw"):
            ee_yaw_for_reward = float(self._yaw)
        reward, success, reward_info = compute_instruction_reward(
            spec=self._instruction_spec,
            ee_pos=ee,
            obj_pos=obj,
            reward_state=self._reward_state,
            action=action,
            ee_yaw=ee_yaw_for_reward,
            gripper_surface_alignment=gripper_surface_alignment,
            camera_alignment=camera_alignment,
            ee_height_above_surface=ee_height_above_surface,
            gripper_command=float(self._last_gripper_cmd),
        )
        caught_body, caught_catalog, caught_score, caught_is_target = self._detect_caught_object(ee_pos=ee)
        if caught_body:
            self._last_caught_body = caught_body
            self._last_caught_catalog = caught_catalog

        self._step_count += 1
        terminated = bool(success)
        truncated = bool(self._step_count >= self.max_steps and not terminated)

        obs = self._get_obs()
        info = self._base_info()
        info.update(reward_info)
        info["success"] = bool(success)
        info["reward"] = float(reward)
        info["step"] = int(self._step_count)
        info["target_grasped"] = bool(float(reward_info.get("grasped", 0.0)) >= 0.5)
        info["caught_object_body"] = caught_body
        info["caught_object_catalog"] = caught_catalog
        info["caught_object_score"] = float(caught_score)
        info["caught_object_is_target"] = bool(caught_is_target)
        info["last_caught_object_body"] = self._last_caught_body
        info["last_caught_object_catalog"] = self._last_caught_catalog
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if self.sim is not None:
            try:
                self.sim.cleanup()
            except Exception:
                pass
        self.sim = None
        self._cleanup_generated_files()
        self._current_wrapper_xml = None
        self._desk_texture_name = ""
        self._prev_object_positions = {}
        self._inverse_catalog_to_body = {}

    def _sample_scene(self, options: Optional[dict[str, Any]]) -> SceneSpec:
        requested_scene = (options or {}).get("scene")
        if requested_scene is not None:
            requested_scene = str(requested_scene)
            for scene in self.scenes:
                if scene.name == requested_scene:
                    return scene
        idx = int(self.np_random.integers(0, len(self.scenes)))
        return self.scenes[idx]

    def _build_wrapper(self, scene: SceneSpec) -> Path:
        build_wrapper_if_needed = _import_wrapper_builder()
        wrapper_out = None
        use_cache = bool(self.use_wrapper_cache)
        if (not use_cache) or self.wrapper_cleanup:
            wrapper_out = self._temporary_wrapper_path(scene=scene)
            use_cache = False
        ee_start = np.asarray(self.defaults.get("ee_start", (0.0, 0.0, MIN_EE_START_Z)), dtype=float).reshape(3)
        ee_start[2] = max(float(ee_start[2]), MIN_EE_START_Z)

        wrapper_xml = build_wrapper_if_needed(
            scene_name=scene.name,
            object_names=list(scene.objects),
            scene_z=self.defaults.get("scene_z", -0.85),
            ee_start=tuple(float(x) for x in ee_start),
            table_z=self.defaults.get("table_z", 0.15),
            settle_time=self.defaults.get("settle_time", 0.0),
            wrapper_out=wrapper_out,
            use_cache=use_cache,
        )
        if self.wrapper_cleanup:
            self._register_cleanup_path(wrapper_xml)

        self._desk_texture_name = ""
        if self.desk_texture_files:
            tex_idx = int(self.np_random.integers(0, len(self.desk_texture_files)))
            chosen_texture = self.desk_texture_files[tex_idx]
            variant_tag = f"rl_{int(time.time_ns())}"
            patched = _build_textured_wrapper_variant(
                base_wrapper_xml=wrapper_xml,
                chosen_texture=chosen_texture,
                variant_tag=variant_tag,
                desk_geom_regex=self.desk_geom_regex,
                desk_texrepeat=self.desk_texrepeat,
            )
            wrapper_xml = patched.wrapper_xml
            self._desk_texture_name = chosen_texture.name

            if self.wrapper_cleanup:
                for path in patched.generated_xmls:
                    self._register_cleanup_path(path)
                for path in patched.generated_files:
                    self._register_cleanup_path(path)

        return wrapper_xml

    def _refresh_workspace_safety(self):
        self._support_surface_z = float(infer_workspace_surface_z(self.sim, fallback_z=0.0))
        ee_bottom = float(body_bottom_offset(self.sim, "ee_base"))
        # Keep tool lowest point >= 5 cm above support; spawn at >= 8 cm.
        self._ee_min_z = self._support_surface_z + ee_bottom + 0.05
        self._ee_spawn_z = self._support_surface_z + ee_bottom + 0.08

    def _move_ee_to_spawn_height(self):
        if self.sim is None:
            return
        ee = self._get_ee_position().astype(np.float64)
        if ee[2] >= self._ee_spawn_z - 1e-4:
            return
        target = ee.copy()
        target[2] = self._ee_spawn_z
        self._set_ee_target(target)
        if hasattr(self.sim, "goto"):
            try:
                self.sim.goto(target, max_steps=120, tol=0.01)
            except Exception:
                pass
        if hasattr(self.sim, "hold_current_pose"):
            try:
                self.sim.hold_current_pose(warm_steps=6)
            except Exception:
                pass

    def _temporary_wrapper_path(self, scene: SceneSpec) -> Path:
        obj_part = "-".join(sorted(scene.objects))
        stamp = int(time.time_ns())
        return self.wrapper_dir / f"{scene.name}__{obj_part}__rltmp_{stamp}.xml"

    def _clear_sim_recording_buffers(self):
        if self.sim is None:
            return
        # Reset any simulator-side logs so reset-time warmup motion never appears in saved episodes.
        for attr in ("trajectory_data", "overview_frames", "ee_camera_frames"):
            if hasattr(self.sim, attr):
                try:
                    setattr(self.sim, attr, [])
                except Exception:
                    pass

    def _register_cleanup_path(self, path: Path):
        p = Path(path).resolve()
        if p in self._cleanup_path_set:
            return
        self._cleanup_path_set.add(p)
        self._cleanup_paths.append(p)

    def _cleanup_generated_files(self):
        if not self.wrapper_cleanup:
            self._cleanup_paths.clear()
            self._cleanup_path_set.clear()
            return
        for p in reversed(self._cleanup_paths):
            try:
                if p.exists() and p.is_file():
                    p.unlink()
            except Exception:
                pass
        self._cleanup_paths.clear()
        self._cleanup_path_set.clear()

    def _detect_caught_object(self, ee_pos: np.ndarray) -> tuple[str, str, float, bool]:
        if not self._object_body_names or self._reward_state is None:
            self._prev_ee_for_catch = np.asarray(ee_pos, dtype=np.float32).copy()
            return "", "", 0.0, False

        gripper_closed = bool(self._reward_state.gripper_closed)
        ee_now = np.asarray(ee_pos, dtype=np.float32)
        ee_step = ee_now - self._prev_ee_for_catch

        best_body = ""
        best_score = 0.0
        best_dist = 1e9
        for body_name in self._object_body_names:
            try:
                obj_now = self._get_body_position(body_name)
            except Exception:
                continue

            obj_prev = self._prev_object_positions.get(body_name, obj_now)
            obj_step = obj_now - obj_prev
            dist = float(np.linalg.norm(ee_now - obj_now))
            contact_score = float(np.exp(-28.0 * dist))
            follow_score = float(np.exp(-35.0 * np.linalg.norm(obj_step - ee_step)))
            score = contact_score * follow_score

            self._prev_object_positions[body_name] = obj_now

            if score > best_score:
                best_score = score
                best_body = body_name
                best_dist = dist

        self._prev_ee_for_catch = ee_now.copy()
        is_caught = bool(gripper_closed and best_score >= 0.30 and best_dist <= 0.09)
        if not is_caught:
            return "", "", float(best_score), False

        catalog_name = self._inverse_catalog_to_body.get(best_body, best_body)
        return (
            best_body,
            catalog_name,
            float(best_score),
            bool(catalog_name == self._target_catalog_name),
        )

    def _resolve_objects(self, catalog_objects: Sequence[str]) -> tuple[dict[str, str], list[str]]:
        mapping: dict[str, str] = {}
        for name in catalog_objects:
            try:
                mapping[name] = resolve_body_name(self.sim, name)
            except Exception:
                continue

        unique_bodies = []
        for body in mapping.values():
            if body not in unique_bodies:
                unique_bodies.append(body)

        if not unique_bodies:
            unique_bodies = self._discover_dynamic_bodies()
            if len(unique_bodies) == len(catalog_objects):
                for cat, body in zip(catalog_objects, unique_bodies):
                    mapping[cat] = body

        return mapping, unique_bodies

    def _discover_dynamic_bodies(self) -> list[str]:
        model = self.sim.model
        bodies: list[str] = []
        for bid in range(model.nbody):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
            if not name:
                continue
            if any(name.startswith(pfx) for pfx in ROBOT_BODY_PREFIXES):
                continue

            jnum = int(model.body_jntnum[bid])
            jadr = int(model.body_jntadr[bid])
            has_free = False
            for i in range(jnum):
                jid = jadr + i
                if model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
                    has_free = True
                    break
            if has_free:
                bodies.append(name)
        return bodies

    def _read_current_yaw(self) -> float:
        if hasattr(self.sim, "get_yaw"):
            try:
                return float(self.sim.get_yaw())
            except Exception:
                return 0.0
        return 0.0

    def _set_ee_target(self, xyz: np.ndarray):
        if hasattr(self.sim, "set_end_effector_target"):
            self.sim.set_end_effector_target(xyz)
        elif hasattr(self.sim, "set_ee_target"):
            self.sim.set_ee_target(xyz)
        elif hasattr(self.sim, "set_target_position"):
            self.sim.set_target_position(xyz)
        else:  # pragma: no cover - depends on runtime simulator API
            raise RuntimeError("Simulator has no end-effector target setter.")

    def _get_ee_position(self) -> np.ndarray:
        if hasattr(self.sim, "get_end_effector_position"):
            return np.asarray(self.sim.get_end_effector_position(), dtype=np.float32)
        raise RuntimeError("Simulator has no get_end_effector_position method.")

    def _get_body_position(self, body_name: str) -> np.ndarray:
        bid = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_BODY, body_name)
        if bid == -1:
            raise RuntimeError(f"Body '{body_name}' not found in MuJoCo model.")
        return np.asarray(self.sim.data.body_xpos[bid], dtype=np.float32).copy()

    def _get_geom_position(self, geom_name: str) -> Optional[np.ndarray]:
        gid = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_GEOM, geom_name)
        if gid == -1:
            return None
        return np.asarray(self.sim.data.geom_xpos[gid], dtype=np.float32).copy()

    def _get_gripper_surface_alignment(self, obj_pos: np.ndarray) -> Optional[float]:
        """
        Alignment for the requirement:
        stick surface ⟂ line(to object)  => line aligns with stick-surface normal.
        We approximate the normal by the left-right finger separation axis.
        """
        left = self._get_geom_position("finger_l_link")
        right = self._get_geom_position("finger_r_link")
        if left is None or right is None:
            # Fallback to tips if link geoms are unavailable.
            left = self._get_geom_position("finger_l_tip")
            right = self._get_geom_position("finger_r_tip")
        if left is None or right is None:
            return None

        surface_normal_xy = right[:2] - left[:2]
        norm_surface = float(np.linalg.norm(surface_normal_xy))
        if norm_surface < 1e-8:
            return None
        surface_normal_xy /= norm_surface

        gripper_center_xy = 0.5 * (right[:2] + left[:2])
        to_obj_xy = np.asarray(obj_pos[:2] - gripper_center_xy, dtype=np.float32)
        norm_obj = float(np.linalg.norm(to_obj_xy))
        if norm_obj < 1e-8:
            return 1.0
        to_obj_xy /= norm_obj

        # Absolute because either finger can face the target.
        return float(np.clip(abs(np.dot(surface_normal_xy, to_obj_xy)), 0.0, 1.0))

    def _get_ee_camera_alignment(self, obj_pos: np.ndarray) -> Optional[float]:
        cam_id = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")
        if cam_id == -1:
            return None

        cam_pos = np.asarray(self.sim.data.cam_xpos[cam_id], dtype=np.float32)
        cam_xmat = np.asarray(self.sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)
        # MuJoCo fixed camera forward direction is local -Z in world frame.
        cam_forward = -cam_xmat[:, 2]
        norm_forward = float(np.linalg.norm(cam_forward))
        if norm_forward < 1e-8:
            return None
        cam_forward /= norm_forward

        to_obj = np.asarray(obj_pos - cam_pos, dtype=np.float32)
        norm_to_obj = float(np.linalg.norm(to_obj))
        if norm_to_obj < 1e-8:
            return 1.0
        to_obj /= norm_to_obj

        return float(np.clip(np.dot(cam_forward, to_obj), 0.0, 1.0))

    def _apply_action(self, action: np.ndarray):
        ee = self._get_ee_position()
        dxyz = action[:3] * self.action_step_xyz
        target = clamp_xyz(ee + dxyz)
        if np.isfinite(self._ee_min_z):
            target[2] = max(float(target[2]), float(self._ee_min_z))
        self._set_ee_target(target.astype(np.float32))

        if hasattr(self.sim, "set_yaw"):
            self._yaw = self._yaw + float(action[3]) * self.action_step_yaw
            try:
                self.sim.set_yaw(self._yaw)
            except Exception:
                pass

        self._last_gripper_cmd = float(action[4])
        if self._last_gripper_cmd >= 0.2 and hasattr(self.sim, "close_gripper"):
            self.sim.close_gripper()
        elif self._last_gripper_cmd <= -0.2 and hasattr(self.sim, "open_gripper"):
            self.sim.open_gripper()

        self.sim.run_simulation_step(capture_frame=self.capture_frames)

    def _get_obs(self) -> dict[str, np.ndarray]:
        ee_pos = self._get_ee_position()
        target_pos = self._get_body_position(self._target_body_name)

        obj_pos = np.zeros((self.max_objects, 3), dtype=np.float32)
        obj_mask = np.zeros((self.max_objects,), dtype=np.float32)
        for i, body_name in enumerate(self._object_body_names[: self.max_objects]):
            try:
                obj_pos[i] = self._get_body_position(body_name)
                obj_mask[i] = 1.0
            except Exception:
                continue

        onehot = instruction_to_onehot(self._instruction_spec)
        goal_direction = self._instruction_spec.direction.astype(np.float32)

        obs = {
            "ee_position": ee_pos.astype(np.float32),
            "target_object_position": target_pos.astype(np.float32),
            "all_object_positions": obj_pos,
            "object_position_mask": obj_mask,
            "instruction_onehot": onehot,
            "goal_direction": goal_direction,
        }
        return obs

    def _base_info(self) -> dict[str, Any]:
        return {
            "scene": self._scene_name,
            "scene_objects": list(self._scene_catalog_objects),
            "allowed_objects": list(self.allowed_objects),
            "target_object_catalog": self._target_catalog_name,
            "target_object_body": self._target_body_name,
            "language_instruction": self._instruction_spec.text,
            "instruction_type": self._instruction_spec.instruction_type,
            "gripper_command": float(self._last_gripper_cmd),
            "desk_texture": self._desk_texture_name,
            "wrapper_xml": str(self._current_wrapper_xml) if self._current_wrapper_xml else "",
            "support_surface_z": float(self._support_surface_z),
            "ee_min_z": float(self._ee_min_z) if np.isfinite(self._ee_min_z) else float("nan"),
            "ee_spawn_z": float(self._ee_spawn_z) if np.isfinite(self._ee_spawn_z) else float("nan"),
        }
