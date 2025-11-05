import numpy as np
import random

# ===== Safety & motion defaults =====
XYZ_BOUNDS = {
    "x": (-0.90, 0.90),
    "y": (-0.90, 0.90),
    "z": (0.05, 0.60),   # never command below 5cm above floor
}

GRIP_RANGE = (0.0, 0.03)

# Tolerances / heights
DEFAULT_TOL = 0.03         # 3 cm tolerance
DEFAULT_SAFETY_Z = 0.35    # "up" height before moving laterally
DEFAULT_GRASP_Z = 0.06     # approach height for picking
DEFAULT_LIFT_Z  = 0.30     # lift height after grasp

# Timing for smooth segments (in seconds)
SEG_T_UP_S      = 1.5
SEG_T_LATERAL_S = 2.0
SEG_T_DOWN_S    = 1.2
SEG_T_PUSH_S    = 2.0
SETTLE_STEPS = 20
FALLBACK_MAX_STEPS = 120

def clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))

def clamp_xyz(xyz):
    return np.array([
        clamp(xyz[0], *XYZ_BOUNDS["x"]),
        clamp(xyz[1], *XYZ_BOUNDS["y"]),
        clamp(xyz[2], *XYZ_BOUNDS["z"]),
    ], dtype=float)

def minjerk(u: float) -> float:
    u = float(max(0.0, min(1.0, u)))
    return u**3 * (10 - 15*u + 6*u*u)

def follow_segment_minjerk(sim, start_xyz, goal_xyz, duration_s, capture_every_n=3):
    start = np.array(start_xyz, dtype=float)
    goal  = clamp_xyz(goal_xyz)
    dt = float(sim.controller.dt) if hasattr(sim, "controller") else 1.0/60.0
    steps = max(1, int(round(duration_s / dt)))
    for k in range(steps):
        u = (k+1) / steps
        s = minjerk(u)
        p = start + s * (goal - start)
        sim.set_target_position(p)
        capture = ((k % capture_every_n) == 0)
        sim.run_simulation_step(capture_frame=capture)

def settle(sim, steps=SETTLE_STEPS):
    for _ in range(int(steps)):
        sim.run_simulation_step(capture_frame=False)

# ---- Waypoint planner (pick) ----
def plan_pick_waypoints(sim, target_xyz, safety_z=DEFAULT_SAFETY_Z, grasp_z=DEFAULT_GRASP_Z):
    cur = sim.get_end_effector_position().copy()
    up1 = cur.copy(); up1[2] = max(cur[2], safety_z)
    above = np.array([target_xyz[0], target_xyz[1], up1[2]], dtype=float)
    down = np.array([target_xyz[0], target_xyz[1], grasp_z], dtype=float)
    waypoints = [clamp_xyz(up1), clamp_xyz(above), clamp_xyz(down)]
    filtered = [waypoints[0]]
    for p in waypoints[1:]:
        if np.linalg.norm(p - filtered[-1]) > 1e-6:
            filtered.append(p)
    return filtered

def script_pick_and_hover(sim,
                          object_body_name="target_object",
                          yaw=0.0,
                          tol=DEFAULT_TOL,
                          safety_z=DEFAULT_SAFETY_Z,
                          grasp_z=DEFAULT_GRASP_Z,
                          lift_z=DEFAULT_LIFT_Z,
                          **kwargs):
    if hasattr(sim, "hold_current_pose"):
        sim.hold_current_pose(warm_steps=20)
    if hasattr(sim, "open_gripper"):
        sim.open_gripper()

    tgt = sim.get_target_position().copy()
    waypoints = plan_pick_waypoints(sim, tgt, safety_z=safety_z, grasp_z=grasp_z)

    cur = sim.get_end_effector_position().copy()
    follow_segment_minjerk(sim, cur, waypoints[0], SEG_T_UP_S)
    sim.set_target_position(waypoints[0]); sim.goto(waypoints[0], max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)

    follow_segment_minjerk(sim, waypoints[0], waypoints[1], SEG_T_LATERAL_S)
    sim.set_target_position(waypoints[1]); sim.goto(waypoints[1], max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)

    follow_segment_minjerk(sim, waypoints[1], waypoints[2], SEG_T_DOWN_S)
    sim.set_target_position(waypoints[2]); sim.goto(waypoints[2], max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)

    if hasattr(sim, "close_gripper"):
        sim.close_gripper(); settle(sim, steps=30)

    lift_goal = np.array([tgt[0], tgt[1], lift_z], dtype=float)
    cur = sim.get_end_effector_position().copy()
    follow_segment_minjerk(sim, cur, lift_goal, SEG_T_UP_S)
    sim.set_target_position(clamp_xyz(lift_goal)); sim.goto(clamp_xyz(lift_goal), max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)

# ---- True push behavior (side approach, gripper open, yaw set for pushing) ----
def _dir_vec(direction: str):
    if direction == "left":   return np.array([-1.0,  0.0, 0.0], dtype=float)
    if direction == "right":  return np.array([ 1.0,  0.0, 0.0], dtype=float)
    if direction == "forward":return np.array([ 0.0,  1.0, 0.0], dtype=float)
    if direction == "back":   return np.array([ 0.0, -1.0, 0.0], dtype=float)
    return np.array([-1.0, 0.0, 0.0], dtype=float)  # default left

def _yaw_for_push(direction: str):
    """
    Align the finger bar as a flat pusher:
    - pushing along ±X → yaw = 0 (fingers across Y)
    - pushing along ±Y → yaw = pi/2 (fingers across X)
    """
    import math
    if direction in ("left", "right"):   return 0.0
    else:                                 return math.pi/2

def script_push(sim,
                direction="left",
                distance=0.20,
                safety_z=DEFAULT_SAFETY_Z,
                approach_z=0.08,
                yaw=0.0,
                tol=DEFAULT_TOL,
                **kwargs):
    """
    Push along a straight line with side approach:
      - up to safety
      - move above a PRE-CONTACT XY offset (opposite the push direction)
      - descend to approach height
      - slide to CONTACT, then push to GOAL (gripper stays OPEN, yaw set for pushing)
    """
    import math

    tgt = sim.get_target_position().copy()
    dvec = _dir_vec(direction)[:2]          # XY push direction
    dvec = dvec / (np.linalg.norm(dvec) + 1e-8)

    # Where to start relative to the object before pushing:
    contact_offset = 0.06    # 6 cm behind the object's push-side
    pre_xy    = tgt[:2] - dvec * contact_offset
    contact_xy= tgt[:2] - dvec * 0.01       # just reach the side
    goal_xy   = tgt[:2] + dvec * float(abs(distance))

    # Set yaw so the finger bar is perpendicular to motion (flat pushing surface)
    push_yaw = _yaw_for_push(direction)
    if hasattr(sim, "set_yaw"):
        sim.set_yaw(push_yaw)

    # Keep gripper open for push
    if hasattr(sim, "open_gripper"):
        sim.open_gripper()

    # 1) Up to safety
    cur = sim.get_end_effector_position().copy()
    up_goal = np.array([cur[0], cur[1], max(cur[2], safety_z)], dtype=float)
    follow_segment_minjerk(sim, cur, up_goal, SEG_T_UP_S)
    sim.set_target_position(clamp_xyz(up_goal)); sim.goto(clamp_xyz(up_goal), max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)

    # 2) Above PRE-CONTACT
    above_pre = np.array([pre_xy[0], pre_xy[1], up_goal[2]], dtype=float)
    follow_segment_minjerk(sim, up_goal, above_pre, SEG_T_LATERAL_S)
    sim.set_target_position(clamp_xyz(above_pre)); sim.goto(clamp_xyz(above_pre), max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)

    # 3) Down to approach height
    pre_pt = np.array([pre_xy[0], pre_xy[1], approach_z], dtype=float)
    follow_segment_minjerk(sim, above_pre, pre_pt, SEG_T_DOWN_S)
    sim.set_target_position(clamp_xyz(pre_pt)); sim.goto(clamp_xyz(pre_pt), max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)

    # 4) Slide to CONTACT (just touch the side)
    contact_pt = np.array([contact_xy[0], contact_xy[1], approach_z], dtype=float)
    follow_segment_minjerk(sim, pre_pt, contact_pt, 0.6)  # short approach
    sim.set_target_position(clamp_xyz(contact_pt)); sim.goto(clamp_xyz(contact_pt), max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim, steps=10)

    # 5) PUSH to GOAL along direction
    goal = np.array([goal_xy[0], goal_xy[1], approach_z], dtype=float)
    follow_segment_minjerk(sim, contact_pt, goal, SEG_T_PUSH_S)
    sim.set_target_position(clamp_xyz(goal)); sim.goto(clamp_xyz(goal), max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)

# ===== Simple non-overlap object placement =====
def _geom_footprint_radius(model, geom_id):
    gtype = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    if gtype == 6:  # box
        r = float(np.linalg.norm(size[:2]))  # diagonal half-length in XY
        return max(0.03, r)
    elif gtype == 4 or gtype == 0:  # cylinder or sphere
        return max(0.03, float(size[0]))
    else:
        return 0.04

def place_objects_non_overlapping(sim, object_body_names, xy_bounds, min_gap=0.02, max_tries=200):
    """
    Randomly place objects by setting their body pose XY at a fixed Z, avoiding XY overlap.
    - xy_bounds = ((xmin, xmax), (ymin, ymax), fixed_z)
    - Uses each body's first geom footprint to estimate radius.
    For single-object episodes, just pass ['target_object'].
    """
    import mujoco as mj
    model, data = sim.model, sim.data
    xmin, xmax = xy_bounds[0]; ymin, ymax = xy_bounds[1]; z_fixed = xy_bounds[2]

    placed = []
    for name in object_body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        geom_ids = np.where(model.geom_bodyid == bid)[0]
        r = 0.05
        if len(geom_ids) > 0:
            r = _geom_footprint_radius(model, int(geom_ids[0]))
        ok = False
        for _ in range(max_tries):
            x = random.uniform(xmin + r, xmax - r)
            y = random.uniform(ymin + r, ymax - r)
            if all((x - px)**2 + (y - py)**2 >= (r + pr + min_gap)**2 for (px, py, pr) in placed):
                # Try free joint first (typical for movable objects)
                j = model.body_jntnum[bid]
                jadr = model.body_jntadr[bid]
                free_found = False
                for k in range(j):
                    jid = jadr + k
                    if model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
                        qadr = model.jnt_qposadr[jid]
                        data.qpos[qadr:qadr+3] = np.array([x, y, z_fixed], dtype=float)
                        free_found = True
                        break
                if not free_found:
                    # fallback: move kinematic body via xpos (rare for LIBERO objects)
                    data.xpos[bid] = np.array([x, y, z_fixed], dtype=float)
                placed.append((x, y, r))
                ok = True
                break
        if not ok:
            raise RuntimeError(f"Could not place object '{name}' without overlap in {max_tries} tries.")
    mj.mj_forward(model, data)
    return [(px, py) for (px, py, _) in placed]
