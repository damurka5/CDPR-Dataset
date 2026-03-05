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
    prev_heading_toward: Optional[float] = None
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


def _heading_toward_target(
    ee_pos: np.ndarray,
    obj_pos: np.ndarray,
    ee_yaw: Optional[float],
) -> float:
    if ee_yaw is None:
        return 0.0

    to_obj_xy = np.asarray(obj_pos[:2] - ee_pos[:2], dtype=np.float32)
    norm = float(np.linalg.norm(to_obj_xy))
    if norm < 1e-8:
        return 0.0

    toward_xy = to_obj_xy / norm
    heading_xy = np.array([np.cos(float(ee_yaw)), np.sin(float(ee_yaw))], dtype=np.float32)
    alignment = float(np.clip(np.dot(heading_xy, toward_xy), -1.0, 1.0))
    return float(max(alignment, 0.0))


def compute_instruction_reward(
    spec: InstructionSpec,
    ee_pos: np.ndarray,
    obj_pos: np.ndarray,
    reward_state: RewardState,
    action: Optional[np.ndarray] = None,
    ee_yaw: Optional[float] = None,
    gripper_command: Optional[float] = None,
    close_command_threshold: float = 0.2,
    open_command_threshold: float = -0.2,
    grasp_dist_threshold: float = 0.05,
    approach_gain: float = 4.0,
    follow_gain: float = 25.0,
    grasp_confidence_threshold: float = 0.75,
    far_distance_threshold: float = 0.10,
    near_zero_action_threshold: float = 0.08,
    idle_penalty_gain: float = 0.30,
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

    heading_toward = _heading_toward_target(ee_pos=ee_pos, obj_pos=obj_pos, ee_yaw=ee_yaw)
    turning_toward = 0.0
    if reward_state.prev_heading_toward is not None:
        turning_toward = float(max(heading_toward - reward_state.prev_heading_toward, 0.0))
    reward_state.prev_heading_toward = heading_toward
    orientation_reward = 0.30 * heading_toward + 0.18 * turning_toward

    motion_action_norm = 0.0
    idle_action_penalty = 0.0
    if action is not None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        motion_dims = min(4, int(action.shape[0]))
        if motion_dims > 0:
            motion_action_norm = float(np.linalg.norm(action[:motion_dims]))
        if ee_obj_dist > far_distance_threshold and motion_action_norm < near_zero_action_threshold:
            far_scale = float(
                np.clip(
                    (ee_obj_dist - far_distance_threshold) / max(far_distance_threshold, 1e-6),
                    0.0,
                    1.0,
                )
            )
            idle_scale = float(
                np.clip(
                    (near_zero_action_threshold - motion_action_norm)
                    / max(near_zero_action_threshold, 1e-6),
                    0.0,
                    1.0,
                )
            )
            idle_action_penalty = float(idle_penalty_gain * far_scale * idle_scale)

    contact = ee_obj_dist <= grasp_dist_threshold
    was_grasped = bool(reward_state.grasped)
    grasp_confidence = (
        (1.0 if reward_state.gripper_closed and contact else 0.0) * (0.5 + 0.5 * follows_ee)
    )
    grasp_now = grasp_confidence >= grasp_confidence_threshold
    reward_state.grasped = reward_state.grasped or grasp_now
    newly_grasped = bool((not was_grasped) and reward_state.grasped)

    lift = float(obj_pos[2] - reward_state.initial_obj_pos[2])
    lift_progress = float(np.clip(lift / max(spec.lift_target, 1e-6), 0.0, 1.0))

    displacement_xy = obj_pos[:2] - reward_state.initial_obj_pos[:2]
    direction_progress, lateral_error = _direction_progress(
        displacement_xy=displacement_xy,
        direction_xy=spec.direction,
        target_displacement=spec.target_displacement,
    )

    move_stage = 0.0
    approach_stage_done = 0.0
    premature_close_penalty = 0.0

    if spec.instruction_type == "pick_up":
        reward = (
            -0.01
            + 0.35 * approach
            + 1.50 * follows_ee
            + 1.80 * float(reward_state.grasped)
            + 3.50 * lift_progress
            + orientation_reward
            - idle_action_penalty
        )
        success = bool(reward_state.grasped and lift_progress >= 0.95)
    else:
        direction_alignment = float(np.exp(-8.0 * lateral_error))
        approach_dist_threshold = max(0.06, grasp_dist_threshold * 1.2)
        approach_stage_done = float(ee_obj_dist <= approach_dist_threshold)

        if (not was_grasped) and approach_stage_done < 0.5:
            move_stage = 1.0
            if reward_state.gripper_closed:
                premature_close_penalty = 0.05
            reward = (
                -0.01
                + 0.95 * approach
                + 0.70 * orientation_reward
                - idle_action_penalty
                - premature_close_penalty
            )
        elif (not reward_state.grasped) or newly_grasped:
            move_stage = 2.0
            reward = (
                -0.01
                + 0.55 * approach
                + 1.55 * grasp_confidence
                + 1.10 * follows_ee
                + 1.35 * float(newly_grasped)
                + 0.35 * orientation_reward
                - 0.35 * idle_action_penalty
            )
        else:
            move_stage = 3.0
            reward = (
                -0.01
                + 0.25 * approach
                + 0.60 * follows_ee
                + 3.40 * direction_progress
                + 0.50 * direction_alignment
                - 0.20 * idle_action_penalty
            )

        success = bool(
            move_stage >= 3.0
            and reward_state.grasped
            and follows_ee >= 0.55
            and direction_progress >= 0.95
        )

    reward_state.prev_ee_pos = ee_pos.copy()
    reward_state.prev_obj_pos = obj_pos.copy()
    reward_state.step_count += 1

    info = {
        "distance_ee_to_object": ee_obj_dist,
        "approach_reward": approach,
        "follow_score": follows_ee,
        "heading_toward_target": heading_toward,
        "turning_toward_target": turning_toward,
        "orientation_reward": orientation_reward,
        "motion_action_norm": motion_action_norm,
        "idle_action_penalty": idle_action_penalty,
        "grasp_confidence": float(grasp_confidence),
        "gripper_closed": float(reward_state.gripper_closed),
        "grasped": float(reward_state.grasped),
        "newly_grasped": float(newly_grasped),
        "lift_progress": lift_progress,
        "direction_progress": direction_progress,
        "lateral_error": lateral_error,
        "move_stage": move_stage,
        "approach_stage_done": approach_stage_done,
        "premature_close_penalty": premature_close_penalty,
    }
    return float(reward), success, info
