import numpy as np
import mujoco as mj

# Keep these aligned with your OFT runner normalization
XYZ_BOUNDS = ((-0.8, 0.8), (-0.8, 0.8), (0.10, 1.20))
GRIP_RANGE = (0.0, 0.06)

def clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))

def norm11(v, lo, hi):
    return 2.0 * ((np.clip(v, lo, hi) - lo) / (hi - lo + 1e-9)) - 1.0

def make_proprio(sim, normalize=True):
    ee = sim.get_end_effector_position().astype(np.float32)
    yaw = float(sim.get_yaw()) if hasattr(sim, "get_yaw") else 0.0
    grip = float(getattr(sim, "get_gripper_opening", lambda: 0.03)())
    if normalize:
        (xlo,xhi),(ylo,yhi),(zlo,zhi) = XYZ_BOUNDS
        g0,g1 = GRIP_RANGE
        state = np.array([
            norm11(ee[0], xlo, xhi),
            norm11(ee[1], ylo, yhi),
            norm11(ee[2], zlo, zhi),
            np.clip(yaw/np.pi, -1, 1),
            0.0, 0.0, 0.0,
            norm11(grip, g0, g1),
        ], dtype=np.float32)
    else:
        state = np.array([ee[0], ee[1], ee[2], yaw, 0,0,0, grip], dtype=np.float32)
    return state

def capture_obs(sim, task_text):
    full_rgb  = sim.capture_frame(sim.overview_cam, "overview")
    wrist_rgb = sim.capture_frame(sim.ee_cam, "ee_camera")
    return {
        "full_image":  np.ascontiguousarray(full_rgb,  dtype=np.uint8),
        "wrist_image": np.ascontiguousarray(wrist_rgb, dtype=np.uint8),
        "state":       make_proprio(sim, normalize=True),
        "task_description": task_text,
    }

def action_abs_7(xyz, yaw, grip):
    return np.array([xyz[0], xyz[1], xyz[2], float(yaw), 0.0, 0.0, float(grip)], dtype=np.float32)

def action_delta_7(dxyz, dyaw, dgrip):
    return np.array([dxyz[0], dxyz[1], dxyz[2], float(dyaw), 0.0, 0.0, float(dgrip)], dtype=np.float32)

def step_to(sim, xyz_target, yaw_target=None, grip_target=None, inner=2):
    if xyz_target is not None: sim.set_target_position(np.array(xyz_target, dtype=float))
    if yaw_target is not None and hasattr(sim, "set_yaw"): sim.set_yaw(float(yaw_target))
    if grip_target is not None and hasattr(sim, "set_gripper"): sim.set_gripper(float(grip_target))
    for _ in range(inner):
        sim.run_simulation_step(capture_frame=True)

def find_body_xy(sim, name_prefix=("p0_", "p1_", "p2_", "p3_", "p4_")):
    # prefers placed objects created by the scene switcher
    for bid in range(sim.model.nbody):
        nm = mj.mj_id2name(sim.model, mj.mjtObj.mjOBJ_BODY, bid)
        if nm and any(nm.startswith(pfx) for pfx in name_prefix):
            return sim.data.xpos[bid, :2].copy(), nm
    try:
        return sim.get_target_position()[:2].copy(), "target_object"
    except Exception:
        return sim.get_end_effector_position()[:2].copy(), "ee"

def script_pick_and_hover(sim, target_xy=None, z_grasp=0.18, z_lift=0.35, yaw=0.0,
                          approach_offset=(0.0, 0.0), task_text="Pick up the object, then hover."):
    logs = []
    if target_xy is None:
        xy, _ = find_body_xy(sim)
    else:
        xy = np.array(target_xy[:2], dtype=float)
    xy = xy + np.array(approach_offset, dtype=float)

    ee = sim.get_end_effector_position().copy()
    yaw_now = float(sim.get_yaw()) if hasattr(sim, "get_yaw") else 0.0
    grip_now = float(getattr(sim, "get_gripper_opening", lambda: 0.03)())

    waypoints = [
        (np.array([xy[0], xy[1], max(ee[2], z_lift)]), yaw, grip_now),
        (np.array([xy[0], xy[1], z_grasp + 0.07]),      yaw, grip_now),
        (np.array([xy[0], xy[1], z_grasp]),             yaw, grip_now),
    ]
    for wp_xyz, wp_yaw, wp_grip in waypoints:
        prev = sim.get_end_effector_position().copy()
        prev_yaw = yaw_now; prev_grip = grip_now
        step_to(sim, wp_xyz, wp_yaw, wp_grip, inner=12)
        obs = capture_obs(sim, task_text)
        dxyz = sim.get_end_effector_position().copy() - prev
        dyaw = (wp_yaw - prev_yaw) if yaw is not None else 0.0
        dgrip= (wp_grip - prev_grip)
        logs.append((obs, action_abs_7(wp_xyz, wp_yaw, wp_grip), action_delta_7(dxyz, dyaw, dgrip)))
        yaw_now = wp_yaw; grip_now = wp_grip

    # close gripper
    prev = sim.get_end_effector_position().copy()
    prev_yaw = yaw_now; prev_grip = grip_now
    step_to(sim, None, yaw, GRIP_RANGE[0], inner=20)
    obs = capture_obs(sim, task_text)
    dxyz = sim.get_end_effector_position().copy() - prev
    dyaw = (yaw - prev_yaw)
    dgrip = (GRIP_RANGE[0] - prev_grip)
    logs.append((obs, action_abs_7(sim.get_end_effector_position(), yaw, GRIP_RANGE[0]),
                 action_delta_7(dxyz, dyaw, dgrip)))
    grip_now = GRIP_RANGE[0]

    # lift
    lift_xyz = np.array([xy[0], xy[1], z_lift])
    prev = sim.get_end_effector_position().copy()
    prev_yaw = yaw_now; prev_grip = grip_now
    step_to(sim, lift_xyz, yaw, grip_now, inner=20)
    obs = capture_obs(sim, task_text)
    dxyz = sim.get_end_effector_position().copy() - prev
    dyaw = (yaw - prev_yaw)
    dgrip = (grip_now - prev_grip)
    logs.append((obs, action_abs_7(lift_xyz, yaw, grip_now), action_delta_7(dxyz, dyaw, dgrip)))

    # small hover move
    hover2 = lift_xyz + np.array([0.04, 0.0, 0.0])
    prev = sim.get_end_effector_position().copy()
    prev_yaw = yaw_now; prev_grip = grip_now
    step_to(sim, hover2, yaw, grip_now, inner=20)
    obs = capture_obs(sim, task_text)
    dxyz = sim.get_end_effector_position().copy() - prev
    dyaw = (yaw - prev_yaw)
    dgrip = (grip_now - prev_grip)
    logs.append((obs, action_abs_7(hover2, yaw, grip_now), action_delta_7(dxyz, dyaw, dgrip)))

    return logs

def script_push(sim, direction="left", distance=0.15, z_contact=0.18, yaw=0.0,
                task_text="Push the object."):
    logs = []
    xy, _ = find_body_xy(sim)
    start_above = np.array([xy[0], xy[1], z_contact + 0.08])
    path = [start_above,
            np.array([xy[0], xy[1], z_contact])]
    if direction == "left":
        path.append(np.array([xy[0]-distance, xy[1], z_contact]))
    elif direction == "right":
        path.append(np.array([xy[0]+distance, xy[1], z_contact]))
    elif direction == "forward":      # +Y
        path.append(np.array([xy[0], xy[1]+distance, z_contact]))
    else:                              # backward (-Y)
        path.append(np.array([xy[0], xy[1]-distance, z_contact]))
    path.append(start_above)  # retract

    grip = GRIP_RANGE[1]  # open
    prev_xyz = None; prev_yaw = yaw; prev_grip = grip
    for k, xyz in enumerate(path):
        if prev_xyz is None:
            prev_xyz = xyz.copy()
        step_to(sim, xyz, yaw, grip, inner=12)
        obs = capture_obs(sim, task_text)
        cur_xyz = sim.get_end_effector_position().copy()
        dxyz = cur_xyz - prev_xyz
        dyaw = 0.0
        dgrip = 0.0
        logs.append((obs, action_abs_7(cur_xyz, yaw, grip), action_delta_7(dxyz, dyaw, dgrip)))
        prev_xyz = cur_xyz
    return logs
