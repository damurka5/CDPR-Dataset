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
# Tune these to slow down or speed up the robot motion.
SEG_T_UP_S      = 1.5
SEG_T_LATERAL_S = 2.0
SEG_T_DOWN_S    = 1.2
SEG_T_PUSH_S    = 2.0     # time spent on the lateral push leg

# Short settling after each leg (simulation steps)
SETTLE_STEPS = 20

# Small fallback budget if we still need to “snap in” to the exact goal
FALLBACK_MAX_STEPS = 120

# ---- utilities ----
def clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))

def clamp_xyz(xyz):
    return np.array([
        clamp(xyz[0], *XYZ_BOUNDS["x"]),
        clamp(xyz[1], *XYZ_BOUNDS["y"]),
        clamp(xyz[2], *XYZ_BOUNDS["z"]),
    ], dtype=float)

def minjerk(u: float) -> float:
    """Smooth step from 0→1 with zero vel/accel at ends."""
    u = float(max(0.0, min(1.0, u)))
    return u**3 * (10 - 15*u + 6*u*u)

def follow_segment_minjerk(sim, start_xyz, goal_xyz, duration_s, capture_every_n=3):
    """
    Time-parameterized smooth segment: we 'ramp' the target position along a
    min-jerk curve so velocity is gentle. No hard goto() here.
    """
    start = np.array(start_xyz, dtype=float)
    goal  = clamp_xyz(goal_xyz)
    dt = float(sim.controller.dt) if hasattr(sim, "controller") else 1.0/60.0
    steps = max(1, int(round(duration_s / dt)))

    for k in range(steps):
        u = (k+1) / steps
        s = minjerk(u)
        p = start + s * (goal - start)
        sim.set_target_position(p)
        # use the usual sim step; skip excessive frame capture to keep videos compact
        capture = ((k % capture_every_n) == 0)
        sim.run_simulation_step(capture_frame=capture)

def settle(sim, steps=SETTLE_STEPS):
    for _ in range(int(steps)):
        sim.run_simulation_step(capture_frame=False)

# ---- Waypoint planner ----
def plan_pick_waypoints(sim, target_xyz, safety_z=DEFAULT_SAFETY_Z, grasp_z=DEFAULT_GRASP_Z):
    """
    Build a conservative waypoint list:
      1) Rise to max(current_z, safety_z)
      2) Lateral move above the target at 'safety_z'
      3) Descend straight down to 'grasp_z' (clamped)
    """
    cur = sim.get_end_effector_position().copy()
    up1 = cur.copy(); up1[2] = max(cur[2], safety_z)
    above = np.array([target_xyz[0], target_xyz[1], up1[2]], dtype=float)
    down = np.array([target_xyz[0], target_xyz[1], grasp_z], dtype=float)
    waypoints = [clamp_xyz(up1), clamp_xyz(above), clamp_xyz(down)]
    # Deduplicate near-identical consecutive points
    filtered = [waypoints[0]]
    for p in waypoints[1:]:
        if np.linalg.norm(p - filtered[-1]) > 1e-6:
            filtered.append(p)
    return filtered

# ---- High-level scripts ----
def script_pick_and_hover(sim,
                          object_body_name="target_object",
                          yaw=0.0,
                          tol=DEFAULT_TOL,
                          safety_z=DEFAULT_SAFETY_Z,
                          grasp_z=DEFAULT_GRASP_Z,
                          lift_z=DEFAULT_LIFT_Z,
                          **kwargs):
    """
    Robust pick with smooth motion:
      - open gripper
      - up → above target → down (min-jerk segments)
      - short fallback goto with small step budget
      - close gripper
      - lift smoothly to 'lift_z'
    """
    if hasattr(sim, "hold_current_pose"):
        sim.hold_current_pose(warm_steps=20)
    if hasattr(sim, "open_gripper"):
        sim.open_gripper()

    tgt = sim.get_target_position().copy()
    waypoints = plan_pick_waypoints(sim, tgt, safety_z=safety_z, grasp_z=grasp_z)

    # Segment 1: up
    cur = sim.get_end_effector_position().copy()
    follow_segment_minjerk(sim, cur, waypoints[0], SEG_T_UP_S)
    # fallback tighten
    sim.set_target_position(waypoints[0])
    sim.goto(waypoints[0], max_steps=FALLBACK_MAX_STEPS, tol=tol)
    settle(sim)

    # Segment 2: lateral above
    follow_segment_minjerk(sim, waypoints[0], waypoints[1], SEG_T_LATERAL_S)
    sim.set_target_position(waypoints[1])
    sim.goto(waypoints[1], max_steps=FALLBACK_MAX_STEPS, tol=tol)
    settle(sim)

    # Segment 3: down
    follow_segment_minjerk(sim, waypoints[1], waypoints[2], SEG_T_DOWN_S)
    sim.set_target_position(waypoints[2])
    sim.goto(waypoints[2], max_steps=FALLBACK_MAX_STEPS, tol=tol)
    settle(sim)

    # Close, then lift
    if hasattr(sim, "close_gripper"):
        sim.close_gripper()
        settle(sim, steps=30)

    lift_goal = np.array([tgt[0], tgt[1], lift_z], dtype=float)
    cur = sim.get_end_effector_position().copy()
    follow_segment_minjerk(sim, cur, lift_goal, SEG_T_UP_S)   # reuse up timing for the lift
    sim.set_target_position(clamp_xyz(lift_goal))
    sim.goto(clamp_xyz(lift_goal), max_steps=FALLBACK_MAX_STEPS, tol=tol)
    settle(sim)

def script_push(sim,
                direction="left",
                distance=0.20,
                safety_z=DEFAULT_SAFETY_Z,
                approach_z=0.08,
                yaw=0.0,
                tol=DEFAULT_TOL,
                **kwargs):
    """
    Push along a straight line with smooth segments:
      - up to safety (smooth)
      - move above object (smooth)
      - descend to approach height (smooth)
      - push laterally (smooth)
      - each leg finishes with short fallback tightening & settle
    """
    tgt = sim.get_target_position().copy()

    if hasattr(sim, "hold_current_pose"):
        sim.hold_current_pose(warm_steps=10)
    if hasattr(sim, "open_gripper"):
        sim.open_gripper()

    # up
    cur = sim.get_end_effector_position().copy()
    up_goal = np.array([cur[0], cur[1], max(cur[2], safety_z)], dtype=float)
    follow_segment_minjerk(sim, cur, up_goal, SEG_T_UP_S)
    sim.set_target_position(clamp_xyz(up_goal))
    sim.goto(clamp_xyz(up_goal), max_steps=FALLBACK_MAX_STEPS, tol=tol)
    settle(sim)

    # above
    above = np.array([tgt[0], tgt[1], up_goal[2]], dtype=float)
    follow_segment_minjerk(sim, up_goal, above, SEG_T_LATERAL_S)
    sim.set_target_position(clamp_xyz(above))
    sim.goto(clamp_xyz(above), max_steps=FALLBACK_MAX_STEPS, tol=tol)
    settle(sim)

    # down to approach
    down = np.array([tgt[0], tgt[1], approach_z], dtype=float)
    follow_segment_minjerk(sim, above, down, SEG_T_DOWN_S)
    sim.set_target_position(clamp_xyz(down))
    sim.goto(clamp_xyz(down), max_steps=FALLBACK_MAX_STEPS, tol=tol)
    settle(sim)

    # push laterally
    dx, dy = 0.0, 0.0
    d = abs(float(distance))
    if direction == "left":
        dx = -d
    elif direction == "right":
        dx =  d
    elif direction == "forward":
        dy =  d
    elif direction == "back":
        dy = -d
    goal = np.array([down[0] + dx, down[1] + dy, down[2]], dtype=float)

    follow_segment_minjerk(sim, down, goal, SEG_T_PUSH_S)
    sim.set_target_position(clamp_xyz(goal))
    sim.goto(clamp_xyz(goal), max_steps=FALLBACK_MAX_STEPS, tol=tol)
    settle(sim)

# ===== Simple non-overlap object placement =====
def _geom_footprint_radius(model, geom_id):
    """Approximate XY footprint radius from geom type/size."""
    gtype = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    # mjtGeom: 6=box, 4=cylinder, 0=sphere. Handle common ones; fallback ~4 cm.
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
