#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np


def iter_npz_files(root: Path):
    for ep_dir in sorted(root.glob("*")):
        if not ep_dir.is_dir():
            continue
        npz_path = ep_dir / "trajectory_data.npz"
        if npz_path.exists():
            yield ep_dir, npz_path


def _is_numeric_array(x):
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)


def _score_action_candidate(name: str, arr: np.ndarray, action_dim: int):
    """
    Heuristic scoring: prefer names containing 'action', and arrays ending with action_dim.
    """
    name_l = name.lower()
    score = 0
    if "action" in name_l:
        score += 10
    if "act" in name_l:
        score += 3
    if arr.ndim >= 2 and arr.shape[-1] == action_dim:
        score += 10
    # prefer [T, D] over weird shapes
    if arr.ndim == 2 and arr.shape[1] == action_dim:
        score += 5
    # penalize tiny arrays
    if arr.size < 10 * action_dim:
        score -= 5
    return score


def _score_proprio_candidate(name: str, arr: np.ndarray, proprio_dim: int):
    name_l = name.lower()
    score = 0
    if "state" in name_l or "proprio" in name_l:
        score += 10
    if arr.ndim >= 2 and arr.shape[-1] == proprio_dim:
        score += 10
    if arr.ndim == 2 and arr.shape[1] == proprio_dim:
        score += 5
    if arr.size < 10 * proprio_dim:
        score -= 5
    return score


def autodetect_key(npz, kind: str, dim: int):
    best = None
    best_score = -10**9
    for k in npz.files:
        try:
            arr = npz[k]
        except ValueError:
            # object arrays (like task_description) will fail if allow_pickle=False
            # or could fail for other reasons; skip them
            continue

        if not _is_numeric_array(arr):
            continue

        if kind == "action":
            s = _score_action_candidate(k, arr, dim)
        else:
            s = _score_proprio_candidate(k, arr, dim)

        if s > best_score:
            best_score = s
            best = k
    return best, best_score



def flatten_to_TxD(arr: np.ndarray, D: int):
    """
    Accepts shapes like:
      [T, D]
      [T, K, D]  (chunks) -> flatten to [T*K, D]
      [K, D]     -> treat K as T
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        if a.size == D:
            return a.reshape(1, D)
        raise ValueError(f"Cannot reshape 1D array of len {a.size} to (*,{D})")

    if a.shape[-1] != D:
        raise ValueError(f"Last dim {a.shape[-1]} != {D}")

    if a.ndim == 2:
        return a.reshape(-1, D)

    # e.g. [T, K, D] or [B, T, K, D] etc -> collapse all but last dim
    return a.reshape(-1, D)


def robust_quantiles(X: np.ndarray, qlo=0.01, qhi=0.99):
    X = np.asarray(X, dtype=np.float64)
    q01 = np.quantile(X, qlo, axis=0)
    q99 = np.quantile(X, qhi, axis=0)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return q01, q99, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--episodes-root",
        type=str,
        required=True,
        help="Folder containing HUMAN_CONTROL episode subfolders (each with trajectory_data.npz).",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output dataset_statistics.json path.",
    )
    ap.add_argument("--dataset-key", type=str, default="cdpr_local")
    ap.add_argument("--action-dim", type=int, default=5)
    ap.add_argument("--proprio-dim", type=int, default=5)

    ap.add_argument(
        "--action-key",
        type=str,
        default=None,
        help="NPZ key to use for actions. If omitted, auto-detect.",
    )
    ap.add_argument(
        "--proprio-key",
        type=str,
        default=None,
        help="NPZ key to use for proprio/state. If omitted, auto-detect (optional).",
    )
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--print-keys-sample", type=int, default=5)

    args = ap.parse_args()

    root = Path(args.episodes_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_actions = []
    all_proprio = []

    n_eps = 0
    n_transitions = 0
    key_printed = 0

    for ep_dir, npz_path in iter_npz_files(root):
        if args.max_episodes is not None and n_eps >= args.max_episodes:
            break

        with np.load(npz_path, allow_pickle=True) as npz:
            if key_printed < args.print_keys_sample:
                print(f"\n[{ep_dir.name}] keys = {npz.files}")
                key_printed += 1

            # ----- Actions -----
            action_key = args.action_key
            if action_key is None:
                action_key, score = autodetect_key(npz, "action", args.action_dim)
                if action_key is None:
                    print(f"⚠️  {ep_dir.name}: could not auto-detect action key; skipping.")
                    continue
            if action_key not in npz.files:
                print(f"⚠️  {ep_dir.name}: action_key='{action_key}' not found; skipping.")
                continue

            try:
                A = flatten_to_TxD(npz[action_key], args.action_dim)
            except Exception as e:
                print(f"⚠️  {ep_dir.name}: failed to read actions from '{action_key}': {e}; skipping.")
                continue

            # ----- Proprio (optional) -----
            proprio_key = args.proprio_key
            P = None
            if proprio_key is None:
                proprio_key, _ = autodetect_key(npz, "proprio", args.proprio_dim)

            if proprio_key in npz.files:
                try:
                    P = flatten_to_TxD(npz[proprio_key], args.proprio_dim)
                except Exception:
                    P = None

            all_actions.append(A)
            if P is not None:
                # Align lengths if they differ a bit
                m = min(len(P), len(A))
                all_proprio.append(P[:m])

            n_eps += 1
            n_transitions += len(A)

    if not all_actions:
        raise SystemExit(
            "No actions found. Rerun with --print-keys-sample 50 and/or pass --action-key <npz_key>."
        )

    actions = np.concatenate(all_actions, axis=0)
    q01, q99, mean, std = robust_quantiles(actions, 0.01, 0.99)

    stats = {
        args.dataset_key: {
            "num_transitions": int(n_transitions),
            "action": {
                "q01": q01.tolist(),
                "q99": q99.tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "mask": [True] * args.action_dim,
            },
        }
    }

    if all_proprio:
        proprio = np.concatenate(all_proprio, axis=0)
        pq01, pq99, pmean, pstd = robust_quantiles(proprio, 0.01, 0.99)
        stats[args.dataset_key]["proprio"] = {
            "q01": pq01.tolist(),
            "q99": pq99.tolist(),
            "mean": pmean.tolist(),
            "std": pstd.tolist(),
            "mask": [True] * args.proprio_dim,
        }

    out_path.write_text(json.dumps(stats, indent=2))
    print("\n✅ Wrote:", out_path)
    print("Dataset key:", args.dataset_key)
    print("Transitions:", n_transitions)
    print("Action q01:", q01)
    print("Action q99:", q99)
    if "proprio" in stats[args.dataset_key]:
        print("Proprio included:", True)
    else:
        print("Proprio included:", False)
        print("  (If you want it, pass --proprio-key <npz_key> once you know the right key.)")


if __name__ == "__main__":
    main()
