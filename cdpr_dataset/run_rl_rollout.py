#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv
from cdpr_dataset.rl_instruction_tasks import INSTRUCTION_TYPES


def parse_args():
    ap = argparse.ArgumentParser(
        prog="Run RL rollouts in CDPR language-conditioned env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--episodes", type=int, default=20, help="Number of episodes to generate.")
    ap.add_argument("--max_steps", type=int, default=150, help="Max steps per episode.")
    ap.add_argument(
        "--policy",
        type=str,
        default="heuristic",
        choices=["random", "heuristic"],
        help="Action source.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Path to cdpr_scene_catalog.yaml. Uses package default when omitted.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "datasets" / "cdpr_rl_rollouts"),
        help="Directory for episode .npz files.",
    )
    ap.add_argument(
        "--instruction_types",
        nargs="*",
        default=None,
        help="Optional subset: pick_up move_left move_right move_top move_bottom.",
    )
    ap.add_argument(
        "--allowed_objects",
        nargs="*",
        default=["ycb_apple", "ycb_pear", "ycb_peach"],
        help="Track/sample objects only from this list.",
    )
    ap.add_argument(
        "--desk_textures_dir",
        type=str,
        required=True,
        help="Directory with desk textures (.png/.jpg/.jpeg) for per-episode randomization.",
    )
    return ap.parse_args()


def _random_action(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(5,)).astype(np.float32)


def _heuristic_action(obs: dict[str, np.ndarray]) -> np.ndarray:
    ee = obs["ee_position"]
    obj = obs["target_object_position"]
    inst_idx = int(np.argmax(obs["instruction_onehot"]))

    to_obj = obj - ee
    dist = float(np.linalg.norm(to_obj))

    action = np.zeros((5,), dtype=np.float32)

    # Move toward object first.
    action[:3] = np.clip(to_obj * 4.0, -1.0, 1.0)

    if INSTRUCTION_TYPES[inst_idx] == "pick_up":
        if dist < 0.06:
            action[4] = 1.0  # close gripper
            action[2] = max(float(action[2]), 0.8)  # lift
        else:
            action[4] = -1.0  # keep open while approaching
        return action

    direction_map = {
        "move_left": np.array([-1.0, 0.0], dtype=np.float32),
        "move_right": np.array([1.0, 0.0], dtype=np.float32),
        "move_top": np.array([0.0, 1.0], dtype=np.float32),
        "move_bottom": np.array([0.0, -1.0], dtype=np.float32),
    }
    inst_name = INSTRUCTION_TYPES[inst_idx]
    direction = direction_map.get(inst_name, np.zeros((2,), dtype=np.float32))

    if dist < 0.08:
        action[4] = 1.0
        action[0] = direction[0] * 0.8
        action[1] = direction[1] * 0.8
        action[2] = 0.2
    else:
        action[4] = -1.0 if dist > 0.12 else 1.0
    return action


def _append_step(buffer: dict[str, list], obs, action, reward, terminated, truncated, info):
    buffer["ee_position"].append(np.asarray(obs["ee_position"], dtype=np.float32))
    buffer["target_object_position"].append(np.asarray(obs["target_object_position"], dtype=np.float32))
    buffer["all_object_positions"].append(np.asarray(obs["all_object_positions"], dtype=np.float32))
    buffer["object_position_mask"].append(np.asarray(obs["object_position_mask"], dtype=np.float32))
    buffer["instruction_onehot"].append(np.asarray(obs["instruction_onehot"], dtype=np.float32))
    buffer["goal_direction"].append(np.asarray(obs["goal_direction"], dtype=np.float32))
    buffer["actions"].append(np.asarray(action, dtype=np.float32))
    buffer["rewards"].append(np.float32(reward))
    buffer["terminated"].append(np.float32(1.0 if terminated else 0.0))
    buffer["truncated"].append(np.float32(1.0 if truncated else 0.0))
    buffer["target_grasped"].append(np.float32(1.0 if info.get("target_grasped", False) else 0.0))
    buffer["caught_object_score"].append(np.float32(info.get("caught_object_score", 0.0)))
    buffer["caught_object_is_target"].append(
        np.float32(1.0 if info.get("caught_object_is_target", False) else 0.0)
    )
    buffer["caught_object_catalog"].append(str(info.get("caught_object_catalog", "")))
    buffer["caught_object_body"].append(str(info.get("caught_object_body", "")))
    buffer["last_caught_object_catalog"].append(str(info.get("last_caught_object_catalog", "")))
    buffer["last_caught_object_body"].append(str(info.get("last_caught_object_body", "")))


def _to_arrays(buffer: dict[str, list]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in buffer.items()}


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = CDPRLanguageRLEnv(
        catalog_path=args.catalog,
        max_steps=args.max_steps,
        instruction_types=args.instruction_types,
        allowed_objects=args.allowed_objects,
        desk_textures_dir=args.desk_textures_dir,
        seed=args.seed,
    )

    successes = 0
    total_steps = 0

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            step = 0
            ep_reward = 0.0

            episode_buffer: dict[str, list] = {
                "ee_position": [],
                "target_object_position": [],
                "all_object_positions": [],
                "object_position_mask": [],
                "instruction_onehot": [],
                "goal_direction": [],
                "actions": [],
                "rewards": [],
                "terminated": [],
                "truncated": [],
                "target_grasped": [],
                "caught_object_score": [],
                "caught_object_is_target": [],
                "caught_object_catalog": [],
                "caught_object_body": [],
                "last_caught_object_catalog": [],
                "last_caught_object_body": [],
            }

            while not done:
                if args.policy == "random":
                    action = _random_action(rng)
                else:
                    action = _heuristic_action(obs)

                next_obs, reward, terminated, truncated, info = env.step(action)
                _append_step(
                    episode_buffer,
                    obs=obs,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )

                obs = next_obs
                step += 1
                ep_reward += float(reward)
                done = bool(terminated or truncated)

            total_steps += step
            if info.get("success", False):
                successes += 1

            arrays = _to_arrays(episode_buffer)
            ep_file = out_dir / f"episode_{ep:05d}.npz"
            np.savez_compressed(
                ep_file,
                **arrays,
                language_instruction=np.asarray(info.get("language_instruction", ""), dtype=object),
                instruction_type=np.asarray(info.get("instruction_type", ""), dtype=object),
                scene=np.asarray(info.get("scene", ""), dtype=object),
                target_object_catalog=np.asarray(info.get("target_object_catalog", ""), dtype=object),
                target_object_body=np.asarray(info.get("target_object_body", ""), dtype=object),
                scene_objects=np.asarray(info.get("scene_objects", []), dtype=object),
                allowed_objects=np.asarray(info.get("allowed_objects", []), dtype=object),
                desk_texture=np.asarray(info.get("desk_texture", ""), dtype=object),
                wrapper_xml=np.asarray(info.get("wrapper_xml", ""), dtype=object),
                success=np.asarray(bool(info.get("success", False)), dtype=np.bool_),
                episode_return=np.asarray(ep_reward, dtype=np.float32),
            )

            print(
                f"[episode {ep:03d}] steps={step:3d} "
                f"return={ep_reward:8.3f} success={bool(info.get('success', False))} "
                f"instruction='{info.get('language_instruction', '')}'"
            )
    finally:
        env.close()

    success_rate = successes / max(args.episodes, 1)
    mean_len = total_steps / max(args.episodes, 1)
    print(
        f"\nSaved {args.episodes} episodes to: {out_dir}\n"
        f"Success rate: {success_rate:.2%}\n"
        f"Mean episode length: {mean_len:.1f} steps"
    )


if __name__ == "__main__":
    main()
