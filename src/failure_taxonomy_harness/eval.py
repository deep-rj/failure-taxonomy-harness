"""Evaluate a trained BC policy via full-episode rollouts (failure_spec.md).

Loads a trained DoorKeyPolicyModel checkpoint and rolls it out greedily
(argmax over predicted action logits) across a set of seeds, reporting the
fraction of episodes that succeed. Success here is failure_spec.md's own
definition -- terminated=True and reward > 0 -- not per-step action
accuracy (bc.py's compute_accuracy), which is only a training-time proxy.

Default seed set is the *eval* seeds (failure_spec.md §4: range(10000,
10050)), disjoint from the demo seeds used for training. This is distinct
from the train/val split inside scripts/train_bc.py, which only exists to
monitor training itself, not to produce this success-rate number.
"""

from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from minigrid.core.world_object import Door
from minigrid.minigrid_env import MiniGridEnv

from failure_taxonomy_harness.bc import DoorKeyPolicyModel, encode_obs
from failure_taxonomy_harness.env import (
    ACTION_NAMES,
    DEFAULT_ENV_ID,
    DEFAULT_MAX_STEPS,
    DIR_NAMES,
    make_env,
)
from failure_taxonomy_harness.seeds import EVAL_SEEDS

DEFAULT_MODEL_PATH = Path("models/bc_policy.pt")


def _find_door_pos(unwrapped: MiniGridEnv) -> tuple[int, int] | None:
    for i, cell in enumerate(unwrapped.grid.grid):
        if isinstance(cell, Door):
            return (i % unwrapped.grid.width, i // unwrapped.grid.width)
    return None


def rollout(
    model: nn.Module,
    seed: int,
    env_id: str = DEFAULT_ENV_ID,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    """Runs one episode with `model` acting greedily from `seed`.

    Records the §8 per-step trajectory (one row per `env.step()` call, no row
    for the initial post-`reset()` state) using the same ground-truth fields
    `scripts/replay_seed.py` reads for hand-verification -- agent_pos,
    agent_dir, carrying, door_locked, door_open -- so the result feeds
    `predicates.compute_predicates()` and `classifier.classify()` directly.
    `action_name`/`agent_dir_name` are human-readable duplicates of the
    numeric `action`/`agent_dir` codes, for manual inspection only -- the
    classifier keys on state transitions, never on these (or the numeric
    codes -- see §6).

    Returns {"success": bool, "steps": list[dict]}. `success` is True iff the
    episode succeeds per failure_spec.md's definition: terminated=True and
    reward > 0.
    """
    model.eval()
    env = make_env(env_id, max_steps)
    try:
        obs, _info = env.reset(seed=seed)
        unwrapped = cast(MiniGridEnv, env.unwrapped)
        door_pos = _find_door_pos(unwrapped)

        steps: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        reward = 0.0
        for step_idx in range(1, max_steps + 1):
            with torch.no_grad():
                encoded_obs = encode_obs(obs).unsqueeze(0)
                action_logits = model(encoded_obs)
                action = int(torch.argmax(action_logits, dim=1).item())

            obs, reward, terminated, truncated, _info = env.step(action)

            door = unwrapped.grid.get(*door_pos) if door_pos is not None else None
            agent_dir = int(unwrapped.agent_dir)
            steps.append(
                {
                    "seed": seed,
                    "step": step_idx,
                    "action": action,
                    "action_name": ACTION_NAMES[action],
                    "agent_pos": tuple(int(v) for v in unwrapped.agent_pos),
                    "agent_dir": agent_dir,
                    "agent_dir_name": DIR_NAMES[agent_dir],
                    "carrying": unwrapped.carrying is not None,
                    "door_locked": bool(door.is_locked) if isinstance(door, Door) else False,
                    "door_open": bool(door.is_open) if isinstance(door, Door) else False,
                    "reward": float(reward),
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )

            if terminated or truncated:
                break

        return {"success": terminated and float(reward) > 0, "steps": steps}
    finally:
        env.close()


def evaluate_success_rate(
    model: nn.Module,
    seeds,
    env_id: str = DEFAULT_ENV_ID,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> float:
    model.eval()
    successes = 0
    for seed in seeds:
        result = rollout(model, seed, env_id, max_steps)
        success = result["success"]
        print(f"seed={seed:5d}  success={success}")
        successes += success
    return successes / len(seeds)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(EVAL_SEEDS))
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args()

    model = DoorKeyPolicyModel()
    state_dict = torch.load(args.model_path, weights_only=True)
    model.load_state_dict(state_dict)
    success_rate = evaluate_success_rate(model, args.seeds, args.env_id, args.max_steps)
    print(f"\nsuccess_rate={success_rate:.2%} over {len(args.seeds)} seeds")


if __name__ == "__main__":
    main()
