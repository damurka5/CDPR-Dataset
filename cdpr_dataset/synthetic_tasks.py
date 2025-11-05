import numpy as np
import random

# ===== Constants =====
# XYZ bounds for safety (clamped). Tune to your workspace.
XYZ_BOUNDS = {
    "x": (-0.90, 0.90),
    "y": (-0.90, 0.90),
    "z": (0.05, 0.60),   # never command below 5cm above floor
}

# Gripper range in meters (opening distance used by your sim helpers)
GRIP_RANGE = (0.0, 0.03)

# Default safety/tolerance parameters
DEFAULT_TOL = 0.03         # 3 cm tolerance
DEFAULT_SAFETY_Z = 0.35    # "up" height before moving laterally
DEFAULT_GRASP_Z = 0.06     # approach height for picking
DEFAULT_LIFT_Z  = 0.30     # lift height after grasp

# ---- utility clamps ----
def clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))

def clamp_xyz(xyz):
    return np.array([
        clamp(xyz[0], *XYZ_BOUNDS["x"]),
        clamp(xyz[1], *XYZ_BOUNDS["y"]),
        clamp(xyz[2], *XYZ_BOUNDS["z"]),
    ], dtype=float)

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
    Execute a robust pick:
      - open gripper
      - up → above target → down  (3 cm tolerance)
      - close gripper
      - lift to 'lift_z'
    Accepts extra kwargs (ignored) to stay compatible with older runners.
    """
    # Ensure controller starts “neutral”
    if hasattr(sim, "hold_current_pose"):
        sim.hold_current_pose(warm_steps=20)

    # Open gripper before approach
    if hasattr(sim, "open_gripper"):
        sim.open_gripper()

    # Plan waypoint path and execute with tolerance
    tgt = sim.get_target_position().copy()
    waypoints = plan_pick_waypoints(sim, tgt, safety_z=safety_z, grasp_z=grasp_z)
    for p in waypoints:
        ok, _ = sim.goto(p, max_steps=600, tol=tol)
        # add a few settling steps for stability
        for _ in range(10):
            sim.run_simulation_step(capture_frame=False)

    # Close, then lift
    if hasattr(sim, "close_gripper"):
        sim.close_gripper()
        for _ in range(30):
            sim.run_simulation_step(capture_frame=False)

    up_after = np.array([tgt[0], tgt[1], lift_z], dtype=float)
    sim.goto(clamp_xyz(up_after), max_steps=600, tol=tol)
    for _ in range(10):
        sim.run_simulation_step(capture_frame=False)

def script_push(sim,
                direction="left",
                distance=0.20,
                safety_z=DEFAULT_SAFETY_Z,
                approach_z=0.08,
                yaw=0.0,
                tol=DEFAULT_TOL,
                **kwargs):
    """
    Push along a straight line at a constant Z:
      - go up to safety
      - move above object
      - descend to low approach height
      - push along +x/-x/+y/-y by 'distance'
    Accepts extra kwargs (ignored) to stay compatible with older runners.
    """
    tgt = sim.get_target_position().copy()

    if hasattr(sim, "hold_current_pose"):
        sim.hold_current_pose(warm_steps=10)
    if hasattr(sim, "open_gripper"):
        sim.open_gripper()

    # up & above
    up1 = np.array([tgt[0], tgt[1], max(sim.get_end_effector_position()[2], safety_z)], dtype=float)
    sim.goto(clamp_xyz(up1), max_steps=600, tol=tol)
    above = np.array([tgt[0], tgt[1], safety_z], dtype=float)
    sim.goto(clamp_xyz(above), max_steps=600, tol=tol)

    # down to approach height (not touching table)
    down = np.array([tgt[0], tgt[1], approach_z], dtype=float)
    sim.goto(clamp_xyz(down), max_steps=600, tol=tol)

    # lateral push
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
    sim.goto(clamp_xyz(goal), max_steps=800, tol=tol)

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
