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
    prev_ee_motion_align: Optional[float] = None
    prev_gripper_surface_align: Optional[float] = None
    prev_camera_align: Optional[float] = None
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
    gripper_surface_alignment: Optional[float] = None,
    camera_alignment: Optional[float] = None,
    ee_height_above_surface: Optional[float] = None,
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
    near_phase_distance: Optional[float] = None,
    orient_gate_distance: float = 0.10,
    min_ee_height_before_reach: float = 0.10,
    z_height_reach_distance: Optional[float] = None,
    z_height_penalty_gain: float = 120.0,
    motion_dir_gate_distance: Optional[float] = None,
    w_motion_dir_pos_far: float = 40.0,
    w_motion_dir_neg_far: float = 60.0,
    w_motion_dir_pos_near: float = 15.0,
    w_motion_dir_neg_near: float = 25.0,
    w_xyz_pos_far: float = 80.0,
    w_xyz_neg_far: float = 120.0,
    w_xyz_pos_near: float = 20.0,
    w_xyz_neg_near: float = 60.0,
    w_orient_far: float = 2.0,
    w_orient_near: float = 5.0,
    w_gripper_orient_far: float = 1.5,
    w_gripper_orient_near: float = 4.0,
    w_camera_orient_far: float = 1.0,
    w_camera_orient_near: float = 3.0,
    w_obj_pos_far: float = 0.0,
    w_obj_neg_far: float = 0.0,
    w_obj_pos_near: float = 250.0,
    w_obj_neg_near: float = 350.0,
    w_lift_pos_far: float = 0.0,
    w_lift_neg_far: float = 0.0,
    w_lift_pos_near: float = 250.0,
    w_lift_neg_near: float = 350.0,
    grasp_bonus: float = 10.0,
    follow_bonus_scale: float = 4.0,
    action_saturation_threshold: float = 0.90,
    action_saturation_penalty_gain: float = 1.5,
    success_bonus: float = 75.0,
) -> tuple[float, bool, dict[str, float]]:
    ee_pos = np.asarray(ee_pos, dtype=np.float32)
    obj_pos = np.asarray(obj_pos, dtype=np.float32)

    if near_phase_distance is None:
        near_phase_distance = float(far_distance_threshold)
    if z_height_reach_distance is None:
        z_height_reach_distance = float(max(0.06, grasp_dist_threshold * 1.2))
    if motion_dir_gate_distance is None:
        motion_dir_gate_distance = float(max(0.06, grasp_dist_threshold * 1.2))

    if gripper_command is not None:
        if gripper_command >= close_command_threshold:
            reward_state.gripper_closed = True
        elif gripper_command <= open_command_threshold:
            reward_state.gripper_closed = False

    prev_ee_obj_dist = float(np.linalg.norm(reward_state.prev_ee_pos - reward_state.prev_obj_pos))
    ee_obj_dist = float(np.linalg.norm(ee_pos - obj_pos))
    approach = float(np.exp(-approach_gain * ee_obj_dist))  # debug-only signal
    distance_delta = float(prev_ee_obj_dist - ee_obj_dist)

    is_far_phase = bool(ee_obj_dist > near_phase_distance)
    if is_far_phase:
        w_xyz_pos = float(w_xyz_pos_far)
        w_xyz_neg = float(w_xyz_neg_far)
        w_orient = float(w_orient_far)
        w_gripper_orient = float(w_gripper_orient_far)
        w_camera_orient = float(w_camera_orient_far)
        w_motion_dir_pos = float(w_motion_dir_pos_far)
        w_motion_dir_neg = float(w_motion_dir_neg_far)
        w_obj_pos = float(w_obj_pos_far)
        w_obj_neg = float(w_obj_neg_far)
        w_lift_pos = float(w_lift_pos_far)
        w_lift_neg = float(w_lift_neg_far)
    else:
        w_xyz_pos = float(w_xyz_pos_near)
        w_xyz_neg = float(w_xyz_neg_near)
        w_orient = float(w_orient_near)
        w_gripper_orient = float(w_gripper_orient_near)
        w_camera_orient = float(w_camera_orient_near)
        w_motion_dir_pos = float(w_motion_dir_pos_near)
        w_motion_dir_neg = float(w_motion_dir_neg_near)
        w_obj_pos = float(w_obj_pos_near)
        w_obj_neg = float(w_obj_neg_near)
        w_lift_pos = float(w_lift_pos_near)
        w_lift_neg = float(w_lift_neg_near)

    distance_improve = float(max(distance_delta, 0.0))
    distance_worsen = float(max(-distance_delta, 0.0))
    xyz_progress_reward = float(w_xyz_pos * distance_improve - w_xyz_neg * distance_worsen)

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

    if gripper_surface_alignment is None:
        gripper_surface_align = heading_toward
    else:
        gripper_surface_align = float(np.clip(gripper_surface_alignment, 0.0, 1.0))
    gripper_surface_turning = 0.0
    if reward_state.prev_gripper_surface_align is not None:
        gripper_surface_turning = float(
            max(gripper_surface_align - reward_state.prev_gripper_surface_align, 0.0)
        )
    reward_state.prev_gripper_surface_align = gripper_surface_align

    if camera_alignment is None:
        camera_align = heading_toward
    else:
        camera_align = float(np.clip(camera_alignment, 0.0, 1.0))
    camera_turning = 0.0
    if reward_state.prev_camera_align is not None:
        camera_turning = float(max(camera_align - reward_state.prev_camera_align, 0.0))
    reward_state.prev_camera_align = camera_align

    orientation_gate = float(ee_obj_dist <= orient_gate_distance)
    heading_orientation_raw = float(heading_toward + 0.5 * turning_toward)
    gripper_orientation_raw = float(gripper_surface_align + 0.5 * gripper_surface_turning)
    camera_orientation_raw = float(camera_align + 0.5 * camera_turning)
    heading_orientation_reward = float(w_orient * orientation_gate * heading_orientation_raw)
    gripper_orientation_reward = float(
        w_gripper_orient * orientation_gate * gripper_orientation_raw
    )
    camera_orientation_reward = float(w_camera_orient * orientation_gate * camera_orientation_raw)
    orientation_raw = float(
        heading_orientation_raw + gripper_orientation_raw + camera_orientation_raw
    )
    orientation_reward = float(
        heading_orientation_reward + gripper_orientation_reward + camera_orientation_reward
    )

    height_below_target = 0.0
    z_height_penalty_active = False
    if ee_height_above_surface is not None and ee_obj_dist > z_height_reach_distance:
        height_below_target = float(max(min_ee_height_before_reach - ee_height_above_surface, 0.0))
        z_height_penalty_active = True
    z_height_penalty = float(z_height_penalty_gain * height_below_target)

    ee_motion_toward = 0.0
    ee_motion_away = 0.0
    ee_motion_alignment = 0.0
    ee_motion_turning = 0.0
    motion_dir_gate = float(ee_obj_dist > motion_dir_gate_distance)
    if motion_dir_gate > 0.0:
        to_obj_prev = obj_pos - reward_state.prev_ee_pos
        to_obj_prev_norm = float(np.linalg.norm(to_obj_prev))
        ee_step_norm = float(np.linalg.norm(ee_step))
        if to_obj_prev_norm > 1e-8 and ee_step_norm > 1e-8:
            to_obj_prev_unit = to_obj_prev / to_obj_prev_norm
            motion_projection = float(np.dot(ee_step, to_obj_prev_unit))
            ee_motion_toward = float(max(motion_projection, 0.0))
            ee_motion_away = float(max(-motion_projection, 0.0))
            ee_motion_alignment = float(np.clip(motion_projection / ee_step_norm, -1.0, 1.0))

    if reward_state.prev_ee_motion_align is not None:
        ee_motion_turning = float(max(ee_motion_alignment - reward_state.prev_ee_motion_align, 0.0))
    reward_state.prev_ee_motion_align = ee_motion_alignment

    motion_dir_reward = float(
        motion_dir_gate
        * (
            w_motion_dir_pos * ee_motion_toward
            - w_motion_dir_neg * ee_motion_away
            + 0.5 * w_motion_dir_pos * ee_motion_turning
        )
    )

    motion_action_norm = 0.0
    idle_action_penalty = 0.0
    action_saturation_penalty = 0.0
    action_saturation_fraction = 0.0
    if action is not None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        motion_dims = min(4, int(action.shape[0]))
        if motion_dims > 0:
            motion_action_norm = float(np.linalg.norm(action[:motion_dims]))
            abs_motion = np.abs(action[:motion_dims])
            sat_over = np.maximum(abs_motion - action_saturation_threshold, 0.0)
            if sat_over.size > 0:
                action_saturation_penalty = float(action_saturation_penalty_gain * np.mean(sat_over))
                action_saturation_fraction = float(
                    np.mean((abs_motion > action_saturation_threshold).astype(np.float32))
                )
        if ee_obj_dist > near_phase_distance and motion_action_norm < near_zero_action_threshold:
            far_scale = float(
                np.clip(
                    (ee_obj_dist - near_phase_distance) / max(near_phase_distance, 1e-6),
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
    lift_step = float(obj_step[2])
    lift_step_reward = float(w_lift_pos * max(lift_step, 0.0) - w_lift_neg * max(-lift_step, 0.0))

    displacement_xy = obj_pos[:2] - reward_state.initial_obj_pos[:2]
    direction_progress, lateral_error = _direction_progress(
        displacement_xy=displacement_xy,
        direction_xy=spec.direction,
        target_displacement=spec.target_displacement,
    )

    obj_motion_in_goal = 0.0
    obj_motion_reward = 0.0
    direction_norm = float(np.linalg.norm(spec.direction))
    if direction_norm > 1e-8:
        goal_dir = np.asarray(spec.direction[:2], dtype=np.float32) / direction_norm
        obj_motion_in_goal = float(np.dot(obj_step[:2], goal_dir))
        obj_motion_reward = float(
            w_obj_pos * max(obj_motion_in_goal, 0.0) - w_obj_neg * max(-obj_motion_in_goal, 0.0)
        )

    move_stage = 1.0 if is_far_phase else 2.0
    approach_stage_done = float(ee_obj_dist <= max(0.06, grasp_dist_threshold * 1.2))
    premature_close_penalty = 0.0

    if spec.instruction_type == "pick_up":
        reward = xyz_progress_reward + orientation_reward + motion_dir_reward
        if not is_far_phase:
            reward += lift_step_reward + follow_bonus_scale * follows_ee + 0.5 * grasp_bonus * grasp_confidence
            move_stage = 3.0 if reward_state.grasped else 2.0
        if newly_grasped:
            reward += grasp_bonus
        reward = (
            reward
            - idle_action_penalty
            - action_saturation_penalty
            - z_height_penalty
        )
        success = bool(reward_state.grasped and lift_progress >= 0.95)
    else:
        reward = xyz_progress_reward + orientation_reward + motion_dir_reward
        if is_far_phase:
            move_stage = 1.0
            if reward_state.gripper_closed:
                premature_close_penalty = 2.0
            reward = (
                reward
                - idle_action_penalty
                - action_saturation_penalty
                - premature_close_penalty
                - z_height_penalty
            )
        else:
            move_stage = 3.0 if reward_state.grasped else 2.0
            reward += obj_motion_reward + follow_bonus_scale * follows_ee + 0.5 * grasp_bonus * grasp_confidence
            if newly_grasped:
                reward += grasp_bonus
            reward = (
                reward
                - 0.5 * idle_action_penalty
                - action_saturation_penalty
                - z_height_penalty
            )

        success = bool(
            move_stage >= 3.0
            and reward_state.grasped
            and follows_ee >= 0.45
            and direction_progress >= 0.95
        )

    if success:
        reward += float(success_bonus)

    reward_state.prev_ee_pos = ee_pos.copy()
    reward_state.prev_obj_pos = obj_pos.copy()
    reward_state.step_count += 1

    info = {
        "distance_ee_to_object": ee_obj_dist,
        "distance_delta": distance_delta,
        "phase_is_far": float(is_far_phase),
        "approach_reward": approach,
        "xyz_progress_reward": xyz_progress_reward,
        "distance_improve": distance_improve,
        "distance_worsen": distance_worsen,
        "follow_score": follows_ee,
        "heading_toward_target": heading_toward,
        "turning_toward_target": turning_toward,
        "gripper_surface_alignment": gripper_surface_align,
        "gripper_surface_turning": gripper_surface_turning,
        "camera_alignment": camera_align,
        "camera_turning": camera_turning,
        "orientation_gate": orientation_gate,
        "heading_orientation_raw": heading_orientation_raw,
        "gripper_orientation_raw": gripper_orientation_raw,
        "camera_orientation_raw": camera_orientation_raw,
        "heading_orientation_reward": heading_orientation_reward,
        "gripper_orientation_reward": gripper_orientation_reward,
        "camera_orientation_reward": camera_orientation_reward,
        "orientation_raw": orientation_raw,
        "orientation_reward": orientation_reward,
        "motion_dir_gate": motion_dir_gate,
        "ee_motion_toward": ee_motion_toward,
        "ee_motion_away": ee_motion_away,
        "ee_motion_alignment": ee_motion_alignment,
        "ee_motion_turning": ee_motion_turning,
        "motion_dir_reward": motion_dir_reward,
        "ee_height_above_surface": (
            float(ee_height_above_surface) if ee_height_above_surface is not None else float("nan")
        ),
        "min_ee_height_before_reach": float(min_ee_height_before_reach),
        "z_height_reach_distance": float(z_height_reach_distance),
        "z_height_penalty_active": float(z_height_penalty_active),
        "z_height_deficit": height_below_target,
        "z_height_penalty": z_height_penalty,
        "motion_action_norm": motion_action_norm,
        "idle_action_penalty": idle_action_penalty,
        "action_saturation_penalty": action_saturation_penalty,
        "action_saturation_fraction": action_saturation_fraction,
        "grasp_confidence": float(grasp_confidence),
        "gripper_closed": float(reward_state.gripper_closed),
        "grasped": float(reward_state.grasped),
        "newly_grasped": float(newly_grasped),
        "lift_progress": lift_progress,
        "lift_step_reward": lift_step_reward,
        "obj_motion_in_goal": obj_motion_in_goal,
        "obj_motion_reward": obj_motion_reward,
        "direction_progress": direction_progress,
        "lateral_error": lateral_error,
        "move_stage": move_stage,
        "approach_stage_done": approach_stage_done,
        "premature_close_penalty": premature_close_penalty,
        "success_bonus": float(success_bonus if success else 0.0),
    }
    return float(reward), success, info
