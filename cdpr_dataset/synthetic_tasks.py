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
DEFAULT_TOL = 0.015         # 1.5 cm tolerance
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

def aabb_of_body(sim, body_name, include_subtree=True):
    """
    World-space AABB (min,max) of all geoms attached to body_name (optionally subtree).
    Returns (xyz_min, xyz_max), each shape (3,).
    """
    import mujoco as mj
    m, d = sim.model, sim.data
    bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1:
        # Helpful message with a few candidate names
        all_names = []
        for i in range(m.nbody):
            try:
                nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i)
            except Exception:
                nm = None
            if nm:
                all_names.append(nm)
        sample = ", ".join(all_names[:20]) + (" ..." if len(all_names) > 20 else "")
        raise ValueError(f"Body '{body_name}' not found in model. Examples: {sample}")

    geoms = []
    if include_subtree:
        # build parent -> children list (once)
        parent = m.body_parentid
        children = {i: [] for i in range(m.nbody)}
        for i in range(1, m.nbody):
            p = parent[i]
            if p >= 0:
                children[p].append(i)
        # DFS
        stack = [bid]; subtree = []
        while stack:
            b = stack.pop()
            subtree.append(b)
            stack.extend(children.get(b, []))
        body_ids = set(subtree)
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in body_ids]
    else:
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]

    if not geoms:
        # fallback to body_xpos
        c = d.body_xpos[bid].copy()
        return c - 1e-3, c + 1e-3

    xyz_min = np.array([ np.inf,  np.inf,  np.inf], dtype=float)
    xyz_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)

    for g in geoms:
        gpos = d.geom_xpos[g]     # world center of the geom frame
        xmat = d.geom_xmat[g].reshape(3,3)  # world rotation of the geom frame
        gtype = m.geom_type[g]
        size  = m.geom_size[g].copy()

        # approximate axis-aligned BB of this geom in world by pushing its local “half-extents”
        # through the rotation and taking abs to get world half-extents.
        # primitives:
        if gtype == 6:  # box
            half = size.copy()  # (x,y,z) half-dims
        elif gtype == 4:  # cylinder
            r, h = size[0], size[1]
            half = np.array([r, r, h], dtype=float)
        elif gtype == 0:  # sphere
            r = size[0]
            half = np.array([r, r, r], dtype=float)
        else:
            # mesh or other: crude sphere bound using first size component
            r = float(size[0]) if size.size > 0 else 0.05
            half = np.array([r, r, r], dtype=float)

        world_half = np.abs(xmat) @ half
        lo = gpos - world_half
        hi = gpos + world_half
        xyz_min = np.minimum(xyz_min, lo)
        xyz_max = np.maximum(xyz_max, hi)

    return xyz_min, xyz_max

def object_centers(sim, body_name="target_object"):
    mn, mx = aabb_of_body(sim, body_name, include_subtree=True)
    center_xy = np.array([(mn[0]+mx[0])*0.5, (mn[1]+mx[1])*0.5], dtype=float)
    top_z     = float(mx[2])
    center_z  = (mn[2]+mx[2])*0.5
    return center_xy, top_z, center_z

def _set_ee(sim, xyz):
    """Set end-effector target only."""
    x = np.asarray(xyz, dtype=float)
    if hasattr(sim, "set_end_effector_target"):
        sim.set_end_effector_target(x)
    elif hasattr(sim, "set_ee_target"):
        sim.set_ee_target(x)
    else:
        sim.set_target_position(x)  # fallback

def follow_segment_minjerk(sim, start_xyz, goal_xyz, duration_s, capture_every_n=2):
    start = np.array(start_xyz, dtype=float)
    goal  = clamp_xyz(goal_xyz)
    dt = float(sim.controller.dt) if hasattr(sim, "controller") else 1.0/60.0
    steps = max(1, int(round(duration_s / dt)))
    for k in range(steps):
        u = (k+1) / steps
        s = minjerk(u)
        p = start + s * (goal - start)
        _set_ee(sim, p)
        capture = ((k % capture_every_n) == 0)
        sim.run_simulation_step(capture_frame=capture)

def settle(sim, steps=SETTLE_STEPS, capture=True):
    for _ in range(int(steps)):
        sim.run_simulation_step(capture_frame=capture)

def log_pick_diagnostics(sim, phase="pre_grasp", object_body_name="object"):
    ee = sim.get_end_effector_position().copy()
    center_xy, top_z, _ = object_centers(sim, object_body_name)
    target = np.array([center_xy[0], center_xy[1], top_z], dtype=float)
    err = np.linalg.norm(ee[:2] - target[:2])
    print(f"[{phase}] EE={ee}  OBJ_TOP={target}  XY_err={err*1000:.1f} mm")


# ---- Waypoint planner (pick) ----
def plan_pick_waypoints(sim, target_xy, top_z,
                        safety_z=DEFAULT_SAFETY_Z,
                        clearance=0.02,      # how far above object to fly
                        grasp_inset=0.002):  # how far into the top to “touch”
    cur = sim.get_end_effector_position().copy()
    up_z = max(cur[2], safety_z, float(top_z) + float(clearance))
    w0 = clamp_xyz([cur[0],         cur[1],         up_z])
    w1 = clamp_xyz([target_xy[0],   target_xy[1],   up_z])
    w2 = clamp_xyz([target_xy[0],   target_xy[1],   float(top_z) + float(grasp_inset)])
    return [w0, w1, w2]

    
def script_pick_and_hover(sim,
                          object_body_name="object",
                          yaw=0.0,
                          tol=DEFAULT_TOL,
                          safety_z=DEFAULT_SAFETY_Z,
                          lift_z=DEFAULT_LIFT_Z,
                          **kwargs):
    # Stabilize + open
    if hasattr(sim, "hold_current_pose"):
        sim.hold_current_pose(warm_steps=20)
    if hasattr(sim, "open_gripper"):
        sim.open_gripper()

    # Compute object top/center
    center_xy, top_z, _ = object_centers(sim, object_body_name)

    # Build waypoints: up → above → down-to-contact (object-relative)
    waypoints = plan_pick_waypoints(sim,
                                    target_xy=center_xy,
                                    top_z=top_z,
                                    safety_z=safety_z,
                                    clearance=0.015,
                                    grasp_inset=0.004)

    # Execute (single min-jerk per leg; no duplicate goto)
    cur = sim.get_end_effector_position().copy()
    follow_segment_minjerk(sim, cur,           waypoints[0], SEG_T_UP_S);       settle(sim, 10)
    follow_segment_minjerk(sim, waypoints[0],  waypoints[1], SEG_T_LATERAL_S);  settle(sim, 10)
    follow_segment_minjerk(sim, waypoints[1],  waypoints[2], SEG_T_DOWN_S);     settle(sim, 15)

    log_pick_diagnostics(sim, phase="pre_grasp", object_body_name=object_body_name)

    # Close + settle
    if hasattr(sim, "close_gripper"):
        sim.close_gripper()
    settle(sim, 30)

    # Lift to a safe height above both safety & object
    lift_goal = clamp_xyz([center_xy[0], center_xy[1], max(lift_z, safety_z, top_z + 0.10)])
    follow_segment_minjerk(sim, sim.get_end_effector_position().copy(), lift_goal, SEG_T_UP_S);  settle(sim, 10)

    log_pick_diagnostics(sim, phase="post_lift", object_body_name=object_body_name)


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
                object_body_name="object",
                direction="left",
                distance=0.20,
                safety_z=DEFAULT_SAFETY_Z,
                approach_z=None,
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

    # tgt = sim.get_target_position().copy()
    center_xy, top_z, _ = object_centers(sim, object_body_name)
    tgt = np.array([center_xy[0], center_xy[1], top_z], dtype=float)

    dvec = _dir_vec(direction)[:2]          # XY push direction
    dvec = dvec / (np.linalg.norm(dvec) + 1e-8)

    # Where to start relative to the object before pushing:
    contact_offset = 0.06    # 6 cm behind the object's push-side
    pre_xy    = tgt[:2] - dvec * contact_offset
    contact_xy= tgt[:2] - dvec * 0.01       # just reach the side
    goal_xy   = tgt[:2] + dvec * float(abs(distance))

    # Push at object side height
    if approach_z is None:
       approach_z = top_z + 0.005
       
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
    follow_segment_minjerk(sim, cur, clamp_xyz(up_goal), SEG_T_UP_S);  settle(sim, 10)

    # 2) Above PRE-CONTACT
    above_pre = np.array([pre_xy[0], pre_xy[1], up_goal[2]], dtype=float)
    follow_segment_minjerk(sim, up_goal, above_pre, SEG_T_LATERAL_S)
    sim.set_target_position(clamp_xyz(above_pre)); sim.goto(clamp_xyz(above_pre), max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)

    # 3) Down to approach height
    pre_pt = np.array([pre_xy[0], pre_xy[1], approach_z], dtype=float)
    if hasattr(sim, "open_gripper"): sim.open_gripper()
    follow_segment_minjerk(sim, clamp_xyz(up_goal), clamp_xyz(above_pre), SEG_T_LATERAL_S);  settle(sim, 10)

    # 4) Slide to CONTACT (just touch the side)
    contact_pt = np.array([contact_xy[0], contact_xy[1], approach_z], dtype=float)
    if hasattr(sim, "open_gripper"): sim.open_gripper()
    follow_segment_minjerk(sim, clamp_xyz(above_pre), clamp_xyz(pre_pt), SEG_T_DOWN_S);  settle(sim, 10)
    
    # 5) PUSH to GOAL along direction
    goal = np.array([goal_xy[0], goal_xy[1], approach_z], dtype=float)
    if hasattr(sim, "open_gripper"): sim.open_gripper()
    follow_segment_minjerk(sim, clamp_xyz(contact_pt), clamp_xyz(goal), SEG_T_PUSH_S);  settle(sim, 10)

def script_move_to_xy(sim,
                      object_body_name="object",
                      goal_xy=(0.0, 0.0),
                      safety_z=DEFAULT_SAFETY_Z,
                      tol=DEFAULT_TOL):
    """
    Pick object from wherever it is and place it so its top center ends up above goal_xy.
    """
    # 1) Compute object top/center
    center_xy, top_z, _ = object_centers(sim, object_body_name)

    # 2) Pick (tight tolerances)
    script_pick_and_hover(sim,
                          object_body_name=object_body_name,
                          tol=0.01,  # stricter than default
                          safety_z=safety_z,
                          grasp_z=top_z + 0.004,
                          lift_z=max(safety_z, top_z + 0.20))

    # 3) Move above goal
    above_goal = np.array([goal_xy[0], goal_xy[1], max(safety_z, top_z + 0.20)], dtype=float)
    cur = sim.get_end_effector_position().copy()
    follow_segment_minjerk(sim, cur, above_goal, SEG_T_LATERAL_S)
    sim.set_target_position(clamp_xyz(above_goal)); sim.goto(clamp_xyz(above_goal), max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)

    # 4) Place: descend to slightly above table then release
    table_z = getattr(sim, "table_z", 0.15)  # if your sim exposes it; else hardcode
    place_z = table_z + 0.01  # 1 cm above table; tune
    down_pt = np.array([goal_xy[0], goal_xy[1], place_z], dtype=float)
    follow_segment_minjerk(sim, above_goal, down_pt, SEG_T_DOWN_S)
    sim.set_target_position(clamp_xyz(down_pt)); sim.goto(clamp_xyz(down_pt), max_steps=FALLBACK_MAX_STEPS, tol=0.008); settle(sim, steps=20)

    if hasattr(sim, "open_gripper"):
        sim.open_gripper(); settle(sim, steps=20)

    # 5) Retract
    up = np.array([goal_xy[0], goal_xy[1], max(safety_z, place_z + 0.15)], dtype=float)
    follow_segment_minjerk(sim, down_pt, up, SEG_T_UP_S)
    sim.set_target_position(clamp_xyz(up)); sim.goto(clamp_xyz(up), max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)


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
