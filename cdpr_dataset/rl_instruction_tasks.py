from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


INSTRUCTION_TYPES: tuple[str, ...] = (
    "pick_up",
    "move_left",
    "move_right",
    "move_top",
    "move_bottom",
)

MOVE_DIRECTIONS: dict[str, np.ndarray] = {
    "move_left": np.array([-1.0, 0.0], dtype=np.float32),
    "move_right": np.array([1.0, 0.0], dtype=np.float32),
    "move_top": np.array([0.0, 1.0], dtype=np.float32),
    "move_bottom": np.array([0.0, -1.0], dtype=np.float32),
}


@dataclass(frozen=True)
class InstructionSpec:
    instruction_type: str
    text: str
    target_object: str
    direction: np.ndarray
    target_displacement: float
    lift_target: float


@dataclass
class RewardState:
    initial_ee_pos: np.ndarray
    initial_obj_pos: np.ndarray
    prev_ee_pos: np.ndarray
    prev_obj_pos: np.ndarray
    gripper_closed: bool = False
    grasped: bool = False
    step_count: int = 0


def canonical_object_name(name: str) -> str:
    return str(name).replace("_", " ").strip()


def instruction_type_to_index(instruction_type: str) -> int:
    try:
        return INSTRUCTION_TYPES.index(instruction_type)
    except ValueError as exc:
        raise KeyError(f"Unknown instruction type: {instruction_type}") from exc


def instruction_to_onehot(spec: InstructionSpec) -> np.ndarray:
    out = np.zeros((len(INSTRUCTION_TYPES),), dtype=np.float32)
    out[instruction_type_to_index(spec.instruction_type)] = 1.0
    return out


def sample_instruction(
    target_object: str,
    rng: np.random.Generator,
    allowed_instruction_types: Optional[Sequence[str]] = None,
    move_distance: float = 0.20,
    lift_distance: float = 0.10,
) -> InstructionSpec:
    if allowed_instruction_types is None:
        candidates = list(INSTRUCTION_TYPES)
    else:
        allowed_set = set(allowed_instruction_types)
        candidates = [t for t in INSTRUCTION_TYPES if t in allowed_set]

    if not candidates:
        raise ValueError("allowed_instruction_types removed all instruction types.")

    instruction_type = candidates[int(rng.integers(0, len(candidates)))]
    nice_obj = canonical_object_name(target_object)

    if instruction_type == "pick_up":
        text = f"pick up {nice_obj}"
        direction = np.zeros((2,), dtype=np.float32)
    elif instruction_type == "move_left":
        text = f"move {nice_obj} to left"
        direction = MOVE_DIRECTIONS[instruction_type]
    elif instruction_type == "move_right":
        text = f"move {nice_obj} to right"
        direction = MOVE_DIRECTIONS[instruction_type]
    elif instruction_type == "move_top":
        text = f"move {nice_obj} to top"
        direction = MOVE_DIRECTIONS[instruction_type]
    elif instruction_type == "move_bottom":
        text = f"move {nice_obj} to bottom"
        direction = MOVE_DIRECTIONS[instruction_type]
    else:
        raise RuntimeError(f"Unsupported sampled instruction: {instruction_type}")

    return InstructionSpec(
        instruction_type=instruction_type,
        text=text,
        target_object=target_object,
        direction=np.asarray(direction, dtype=np.float32),
        target_displacement=float(move_distance),
        lift_target=float(lift_distance),
    )


def init_reward_state(initial_ee_pos: np.ndarray, initial_obj_pos: np.ndarray) -> RewardState:
    initial_ee_pos = np.asarray(initial_ee_pos, dtype=np.float32).copy()
    initial_obj_pos = np.asarray(initial_obj_pos, dtype=np.float32).copy()
    return RewardState(
        initial_ee_pos=initial_ee_pos,
        initial_obj_pos=initial_obj_pos,
        prev_ee_pos=initial_ee_pos.copy(),
        prev_obj_pos=initial_obj_pos.copy(),
    )


def _direction_progress(
    displacement_xy: np.ndarray, direction_xy: np.ndarray, target_displacement: float
) -> tuple[float, float]:
    norm = float(np.linalg.norm(direction_xy))
    if norm < 1e-8:
        return 0.0, float(np.linalg.norm(displacement_xy))

    unit = direction_xy / norm
    proj = float(np.dot(displacement_xy, unit))
    progress = float(np.clip(proj / max(target_displacement, 1e-6), 0.0, 1.0))
    lateral = displacement_xy - unit * proj
    lateral_error = float(np.linalg.norm(lateral))
    return progress, lateral_error


def compute_instruction_reward(
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    obj_pos: np.ndarray,
    reward_state: RewardState,
    gripper_command: Optional[float] = None,
    close_command_threshold: float = 0.2,
    open_command_threshold: float = -0.2,
    grasp_dist_threshold: float = 0.05,
    approach_gain: float = 4.0,
    follow_gain: float = 25.0,
    grasp_confidence_threshold: float = 0.75,
) -> tuple[float, bool, dict[str, float]]:
    ee_pos = np.asarray(ee_pos, dtype=np.float32)
    obj_pos = np.asarray(obj_pos, dtype=np.float32)

    if gripper_command is not None:
        if gripper_command >= close_command_threshold:
            reward_state.gripper_closed = True
        elif gripper_command <= open_command_threshold:
            reward_state.gripper_closed = False

    ee_obj_dist = float(np.linalg.norm(ee_pos - obj_pos))
    approach = float(np.exp(-approach_gain * ee_obj_dist))

    obj_step = obj_pos - reward_state.prev_obj_pos
    ee_step = ee_pos - reward_state.prev_ee_pos
    follow_error = float(np.linalg.norm(obj_step - ee_step))

    follows_ee = 0.0
    if reward_state.gripper_closed and ee_obj_dist <= (grasp_dist_threshold * 1.6):
        follows_ee = float(np.exp(-follow_gain * follow_error))

    contact = ee_obj_dist <= grasp_dist_threshold
    grasp_confidence = (
        (1.0 if reward_state.gripper_closed and contact else 0.0) * (0.5 + 0.5 * follows_ee)
    )
    grasp_now = grasp_confidence >= grasp_confidence_threshold
    reward_state.grasped = reward_state.grasped or grasp_now

    lift = float(obj_pos[2] - reward_state.initial_obj_pos[2])
    lift_progress = float(np.clip(lift / max(spec.lift_target, 1e-6), 0.0, 1.0))

    displacement_xy = obj_pos[:2] - reward_state.initial_obj_pos[:2]
    direction_progress, lateral_error = _direction_progress(
        displacement_xy=displacement_xy,
        direction_xy=spec.direction,
        target_displacement=spec.target_displacement,
    )

    if spec.instruction_type == "pick_up":
        reward = (
            -0.01
            + 0.35 * approach
            + 1.50 * follows_ee
            + 1.80 * float(reward_state.grasped)
            + 3.50 * lift_progress
        )
        success = bool(reward_state.grasped and lift_progress >= 0.95)
    else:
        direction_alignment = float(np.exp(-8.0 * lateral_error))
        reward = (
            -0.01
            + 0.25 * approach
            + 1.20 * float(reward_state.grasped)
            + 0.90 * follows_ee
            + 3.20 * direction_progress
            + 0.35 * direction_alignment
        )
        success = bool(direction_progress >= 0.95 and (reward_state.grasped or follows_ee >= 0.55))

    reward_state.prev_ee_pos = ee_pos.copy()
    reward_state.prev_obj_pos = obj_pos.copy()
    reward_state.step_count += 1

    info = {
        "distance_ee_to_object": ee_obj_dist,
        "approach_reward": approach,
        "follow_score": follows_ee,
        "grasp_confidence": float(grasp_confidence),
        "gripper_closed": float(reward_state.gripper_closed),
        "grasped": float(reward_state.grasped),
        "lift_progress": lift_progress,
        "direction_progress": direction_progress,
        "lateral_error": lateral_error,
    }
    return float(reward), success, info
