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
DEFAULT_TOL = 0.01         # 1.5 cm tolerance
DEFAULT_SAFETY_Z = 0.35    # "up" height before moving laterally
DEFAULT_GRASP_Z = 0.06     # approach height for picking
DEFAULT_LIFT_Z  = 0.30     # lift height after grasp

# Timing for smooth segments (in seconds)
SEG_T_UP_S      = 1.0 # was 1.5
SEG_T_LATERAL_S = 1.2 # was 2.0 
SEG_T_DOWN_S    = 0.5 # was 1.2  
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

def _goto_if_available(sim, target_xyz, tol=DEFAULT_TOL):
    target = clamp_xyz(target_xyz)
    # seed whichever API exists
    if hasattr(sim, "set_end_effector_target"):
        sim.set_end_effector_target(target)
    elif hasattr(sim, "set_ee_target"):
        sim.set_ee_target(target)
    elif hasattr(sim, "set_target_position"):
        sim.set_target_position(target)
    # refine using feedback
    if hasattr(sim, "goto"):
        sim.goto(target, max_steps=FALLBACK_MAX_STEPS, tol=tol)
    settle(sim, 10)

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

def resolve_body_name(sim, logical_name: str) -> str:
    """
    Resolve a logical body name to an actual MuJoCo body name.
    - First tries exact match.
    - Then falls back to substring search (useful for prefixed LIBERO objects like 'p0_red_bowl').
    """
    import mujoco as mj
    m = sim.model

    # exact match
    bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, logical_name)
    if bid != -1:
        return logical_name

    # substring fallback
    for i in range(m.nbody):
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i)
        if nm and logical_name in nm:
            return nm

    raise ValueError(f"Could not resolve body name for '{logical_name}'.")

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

def task_language(task_name: str, object_name: str) -> str:
    """
    Map internal task_name + catalog object_name to a natural language instruction.
    object_name should be the catalog name (e.g., 'milk', 'ketchup').
    """
    nice_obj = object_name.replace("_", " ")

    if task_name == "pick_and_hover":
        return f"pick up the {nice_obj} and hover above the table"

    if task_name == "move_to_center":
        return f"pick up the {nice_obj} and move it to the center of the table"

    if task_name == "push_left":
        return f"push the {nice_obj} to the left"
    if task_name == "push_right":
        return f"push the {nice_obj} to the right"
    if task_name == "push_forward":
        return f"push the {nice_obj} forward"
    if task_name == "push_back":
        return f"push the {nice_obj} back"

    if task_name == "put_into_bowl":
        # return f"put the {nice_obj} into the bowl"
        return f"put {nice_obj} on plate"

    # fallback: at least something
    return f"do the {task_name} task with the {nice_obj}"
    
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
                                    clearance=0.00,
                                    grasp_inset=0.002)

    # Execute (single min-jerk per leg; no duplicate goto)
    cur = sim.get_end_effector_position().copy()
    # Up
    follow_segment_minjerk(sim, cur, waypoints[0], SEG_T_UP_S)
    _goto_if_available(sim, waypoints[0], tol=tol)

    # Lateral
    follow_segment_minjerk(sim, waypoints[0], waypoints[1], SEG_T_LATERAL_S)
    _goto_if_available(sim, waypoints[1], tol=tol)

    # Down
    follow_segment_minjerk(sim, waypoints[1], waypoints[2], SEG_T_DOWN_S)
    _goto_if_available(sim, waypoints[2], tol=tol)

    log_pick_diagnostics(sim, phase="pre_grasp", object_body_name=object_body_name)

    # Close + settle
    if hasattr(sim, "close_gripper"):
        sim.close_gripper()
    settle(sim, 30)

    # Lift to a safe height above both safety & object
    lift_goal = clamp_xyz([center_xy[0], center_xy[1], max(lift_z, safety_z, top_z + 0.10)])
    follow_segment_minjerk(sim, sim.get_end_effector_position().copy(), lift_goal, SEG_T_UP_S);  settle(sim, 10)
    _goto_if_available(sim, lift_goal, tol=tol)
        
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
                          lift_z=max(safety_z, top_z))

    # 3) Move above goal
    above_goal = np.array([goal_xy[0], goal_xy[1], max(safety_z, top_z + 0.20)], dtype=float)
    cur = sim.get_end_effector_position().copy()
    follow_segment_minjerk(sim, cur, above_goal, SEG_T_LATERAL_S)
    sim.set_target_position(clamp_xyz(above_goal)); sim.goto(clamp_xyz(above_goal), max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)

    # 4) Place: descend to slightly above table then release
    table_z = getattr(sim, "table_z", 0.15)  # if your sim exposes it; else hardcode
    place_z = table_z + 0.05  # 1 cm above table; tune
    down_pt = np.array([goal_xy[0], goal_xy[1], place_z], dtype=float)
    follow_segment_minjerk(sim, above_goal, down_pt, SEG_T_DOWN_S)
    sim.set_target_position(clamp_xyz(down_pt)); sim.goto(clamp_xyz(down_pt), max_steps=FALLBACK_MAX_STEPS, tol=0.008); settle(sim, steps=20)

    if hasattr(sim, "open_gripper"):
        sim.open_gripper(); settle(sim, steps=20)

    # 5) Retract
    up = np.array([goal_xy[0], goal_xy[1], max(safety_z, place_z + 0.15)], dtype=float)
    follow_segment_minjerk(sim, down_pt, up, SEG_T_UP_S)
    sim.set_target_position(clamp_xyz(up)); sim.goto(clamp_xyz(up), max_steps=FALLBACK_MAX_STEPS, tol=tol); settle(sim)

def script_put_into_bowl(sim,
                         object_body_name="object",
                         bowl_body_name="plate", # change to plate, was "red_bowl"
                         safety_z=DEFAULT_SAFETY_Z,
                         tol=DEFAULT_TOL):
    """
    1) Pick up the object.
    2) Move above the bowl.
    3) Lower into / just above the bowl interior.
    4) Open gripper and retract.
    object_body_name and bowl_body_name can be logical names; we resolve
    them to actual MuJoCo body names (e.g., 'p0_red_bowl').
    """
    # Resolve actual body names (handles prefixes like p0_, p1_, etc.)
    obj_body  = resolve_body_name(sim, object_body_name)
    bowl_body = resolve_body_name(sim, bowl_body_name)

    # 1) Pick the object and hover at a safe height
    script_pick_and_hover(sim,
                          object_body_name=obj_body,
                          tol=tol,
                          safety_z=safety_z,
                          lift_z=max(DEFAULT_LIFT_Z, safety_z))

    # 2) Get bowl pose
    bowl_xy, bowl_top_z, bowl_center_z = object_centers(sim, bowl_body)

    # Move above the bowl
    cur = sim.get_end_effector_position().copy()
    above_z = max(safety_z, bowl_top_z + 0.15)
    above_bowl = np.array([bowl_xy[0], bowl_xy[1], above_z], dtype=float)

    follow_segment_minjerk(sim, cur, above_bowl, SEG_T_LATERAL_S)
    _goto_if_available(sim, above_bowl, tol=tol)

    # 3) Descend into / just above bowl
    # Heuristic: place slightly above bowl center or table, whichever is higher.
    table_z = getattr(sim, "table_z", 0.15)
    place_z = max(table_z + 0.02, bowl_center_z)
    # Don't go above bowl_top_z + 1cm, or we risk dropping on the rim
    place_z = min(place_z, bowl_top_z + 0.01)
    place_z += 0.10

    drop_pt = np.array([bowl_xy[0], bowl_xy[1], place_z], dtype=float)
    follow_segment_minjerk(sim, above_bowl, drop_pt, SEG_T_DOWN_S)
    _goto_if_available(sim, drop_pt, tol=tol)

    # Open gripper to drop object into the bowl
    if hasattr(sim, "open_gripper"):
        sim.open_gripper()
    settle(sim, steps=10)

    # 4) Retract upwards
    retract = np.array([bowl_xy[0], bowl_xy[1], above_z], dtype=float)
    follow_segment_minjerk(sim, drop_pt, retract, SEG_T_UP_S)
    _goto_if_available(sim, retract, tol=tol)
    settle(sim, steps=10)

# ===== Simple non-overlap object placement =====
def _geom_footprint_radius(model, geom_id):
    gtype = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    try:
        import mujoco as mj
        g_box = int(mj.mjtGeom.mjGEOM_BOX)
        g_cylinder = int(mj.mjtGeom.mjGEOM_CYLINDER)
        g_sphere = int(mj.mjtGeom.mjGEOM_SPHERE)
    except Exception:
        g_box, g_cylinder, g_sphere = 6, 4, 0

    if gtype == g_box:  # box
        r = float(np.linalg.norm(size[:2]))  # diagonal half-length in XY
        return max(0.03, r)
    elif gtype == g_cylinder or gtype == g_sphere:  # cylinder or sphere
        return max(0.03, float(size[0]))
    else:
        return 0.04

def _body_has_free_joint(model, bid):
    try:
        import mujoco as mj
        free_type = int(mj.mjtJoint.mjJNT_FREE)
    except Exception:
        free_type = 0
    jn = int(model.body_jntnum[bid])
    ja = int(model.body_jntadr[bid])
    for k in range(jn):
        jid = ja + k
        if model.jnt_type[jid] == free_type:
            return True
    return False

def _geom_world_half_extents(model, data, gid):
    try:
        import mujoco as mj
        g_box = int(mj.mjtGeom.mjGEOM_BOX)
        g_cylinder = int(mj.mjtGeom.mjGEOM_CYLINDER)
        g_capsule = int(mj.mjtGeom.mjGEOM_CAPSULE)
        g_sphere = int(mj.mjtGeom.mjGEOM_SPHERE)
    except Exception:
        g_box, g_cylinder, g_capsule, g_sphere = 6, 4, 3, 0

    gtype = int(model.geom_type[gid])
    size = model.geom_size[gid]
    xmat = data.geom_xmat[gid].reshape(3, 3)

    if gtype == g_box:      # box
        half_local = np.array([size[0], size[1], size[2]], dtype=float)
    elif gtype == g_cylinder:    # cylinder
        half_local = np.array([size[0], size[0], size[1]], dtype=float)
    elif gtype == g_capsule:    # capsule
        half_local = np.array([size[0], size[0], size[1] + size[0]], dtype=float)
    elif gtype == g_sphere:    # sphere
        half_local = np.array([size[0], size[0], size[0]], dtype=float)
    else:
        # Fallback radius bound for meshes/others.
        r = float(model.geom_rbound[gid])
        half_local = np.array([r, r, r], dtype=float)

    return np.abs(xmat) @ half_local

def _infer_workspace_surface_z(sim, fallback_z=0.0):
    """
    Estimate the support surface height in world Z.
    Priority:
    1) Table/desk-like static slabs (highest top-z).
    2) Geom named like floor/ground (plane z).
    3) Fallback.
    """
    import mujoco as mj

    model, data = sim.model, sim.data
    try:
        g_box = int(mj.mjtGeom.mjGEOM_BOX)
        g_plane = int(mj.mjtGeom.mjGEOM_PLANE)
    except Exception:
        g_box, g_plane = 6, 7
    table_top_z = None
    floor_z = None

    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
        gname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid) or ""

        # Ignore dynamic bodies and CDPR robot internals for support detection.
        if _body_has_free_joint(model, bid):
            continue
        if (
            bname.startswith("rotor_")
            or bname.startswith("slider_")
            or bname.startswith("ee_")
            or bname.startswith("camera_")
            or bname.startswith("finger_")
            or bname == "yaw_frame"
            or bname == "ee_platform"
        ):
            continue

        gtype = int(model.geom_type[gid])
        gpos = data.geom_xpos[gid]

        # Plane geoms often encode the ground directly.
        if gtype == g_plane and ("floor" in gname.lower() or "ground" in gname.lower()):
            z = float(gpos[2])
            floor_z = z if floor_z is None else max(floor_z, z)
            continue

        half_world = _geom_world_half_extents(model, data, gid)
        top_z = float(gpos[2] + half_world[2])
        size = model.geom_size[gid]

        # Detect large, flat slabs (desk/table tops) even when unnamed.
        if gtype == g_box:
            dims = sorted([float(size[0]), float(size[1]), float(size[2])])
            is_flat_slab = dims[0] < 0.10 and dims[1] > 0.20 and dims[2] > 0.20
            has_table_name = any(k in gname.lower() for k in ("desk", "table", "counter", "surface"))
            if is_flat_slab or has_table_name:
                table_top_z = top_z if table_top_z is None else max(table_top_z, top_z)

    if table_top_z is not None:
        return table_top_z
    if floor_z is not None:
        return floor_z

    sim_table_z = getattr(sim, "table_z", None)
    if sim_table_z is not None:
        try:
            return float(sim_table_z)
        except Exception:
            pass
    return float(fallback_z)

def _body_bottom_offset(sim, body_name):
    """
    Distance from body origin z to the lowest geom point in its subtree.
    """
    import mujoco as mj

    model, data = sim.model, sim.data
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1:
        return 0.0
    mn, _ = aabb_of_body(sim, body_name, include_subtree=True)
    body_z = float(data.xpos[bid][2])
    return max(0.0, body_z - float(mn[2]))

def infer_workspace_surface_z(sim, fallback_z=0.0):
    return _infer_workspace_surface_z(sim, fallback_z=fallback_z)

def body_bottom_offset(sim, body_name):
    return _body_bottom_offset(sim, body_name)

def place_objects_non_overlapping(
    sim,
    object_body_names,
    xy_bounds,
    min_gap=0.02,
    max_tries=200,
    min_ee_dist=0.08,
    support_clearance=0.002,
):
    """
    Randomly place objects by setting their body pose XY while grounding each object on
    the inferred support surface (table/floor), avoiding XY overlap,
    and avoiding spawning under the end-effector.
    - xy_bounds = ((xmin, xmax), (ymin, ymax), z_hint)
    - Uses each body's first geom footprint to estimate radius.
    For single-object episodes, just pass ['target_object'].
    """
    import mujoco as mj
    model, data = sim.model, sim.data
    xmin, xmax = xy_bounds[0]
    ymin, ymax = xy_bounds[1]
    z_hint = xy_bounds[2]
    surface_z = _infer_workspace_surface_z(sim, fallback_z=z_hint)

    # Avoid placing objects too close to the current EE XY.
    min_ee_dist = float(min_ee_dist)

    if hasattr(sim, "get_end_effector_position"):
        ee = sim.get_end_effector_position().copy()
        ee_x, ee_y = float(ee[0]), float(ee[1])
    else:
        # fallback: assume EE at origin if we can't query it
        ee_x, ee_y = 0.0, 0.0

    def far_from_ee(x, y, min_dist=min_ee_dist):
        return (x - ee_x) ** 2 + (y - ee_y) ** 2 >= (min_dist ** 2)

    placed = []
    for name in object_body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            raise RuntimeError(f"Body '{name}' not found for placement.")
        geom_ids = np.where(model.geom_bodyid == bid)[0]
        r = 0.05
        if len(geom_ids) > 0:
            r = _geom_footprint_radius(model, int(geom_ids[0]))
        z_offset = _body_bottom_offset(sim, name)
        z_target = float(surface_z + z_offset + float(support_clearance))
        ok = False

        for _ in range(max_tries):
            x = random.uniform(xmin + r, xmax - r)
            y = random.uniform(ymin + r, ymax - r)

            # NEW: skip positions too close to EE
            if not far_from_ee(x, y):
                continue

            # Existing non-overlap condition with already placed objects
            if all((x - px)**2 + (y - py)**2 >= (r + pr + min_gap)**2
                   for (px, py, pr) in placed):

                # Try free joint first (typical for movable objects)
                j = model.body_jntnum[bid]
                jadr = model.body_jntadr[bid]
                free_found = False
                for k in range(j):
                    jid = jadr + k
                    if model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
                        qadr = model.jnt_qposadr[jid]
                        data.qpos[qadr:qadr+3] = np.array([x, y, z_target], dtype=float)
                        free_found = True
                        break
                if not free_found:
                    # fallback: move kinematic body via xpos (rare for LIBERO objects)
                    data.xpos[bid] = np.array([x, y, z_target], dtype=float)

                placed.append((x, y, r))
                ok = True
                break

        if not ok:
            raise RuntimeError(f"Could not place object '{name}' without overlap in {max_tries} tries.")

    mj.mj_forward(model, data)
    return [(px, py) for (px, py, _) in placed]
