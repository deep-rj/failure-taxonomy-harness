"""Drive the real env for a seed + action list, printing per-step ground truth.

Used to hand-verify the synthetic trajectories in tests/test_predicates.py against
a real seed rather than an assumed layout -- see failure_spec.md section 4 on
why hand-authored fixtures should be replay-checkable. Usage:

    uv run python scripts/replay_seed.py <seed> <action> <action> ...

Actions are MiniGrid ints: 0=left 1=right 2=forward 3=pickup 4=drop 5=toggle 6=done.
"""

import sys
from typing import cast

from minigrid.core.world_object import Door
from minigrid.minigrid_env import MiniGridEnv

from failure_taxonomy_harness.env import make_env

ACTION_NAMES = ["LEFT", "RIGHT", "FORWARD", "PICKUP", "DROP", "TOGGLE", "DONE"]


def replay(seed: int, actions: list[int], max_steps: int = 50) -> None:
    env = make_env(max_steps=max_steps)
    env.reset(seed=seed)
    unwrapped = cast(MiniGridEnv, env.unwrapped)

    door_pos = None
    for i, cell in enumerate(unwrapped.grid.grid):
        if isinstance(cell, Door):
            door_pos = (i % unwrapped.grid.width, i // unwrapped.grid.width)

    print(
        f"seed={seed} start_pos={tuple(int(v) for v in unwrapped.agent_pos)} "
        f"start_dir={int(unwrapped.agent_dir)} door_pos={door_pos}"
    )

    for step_idx, action in enumerate(actions, start=1):
        _obs, reward, terminated, truncated, _info = env.step(action)
        pos = tuple(int(v) for v in unwrapped.agent_pos)
        direction = int(unwrapped.agent_dir)
        carrying = unwrapped.carrying is not None
        door = unwrapped.grid.get(*door_pos) if door_pos is not None else None
        door_locked = bool(door.is_locked) if isinstance(door, Door) else False
        door_open = bool(door.is_open) if isinstance(door, Door) else False
        print(
            f"step={step_idx:2d} action={ACTION_NAMES[action]:8s} pos={pos} dir={direction} "
            f"carrying={carrying} door_locked={door_locked} door_open={door_open} "
            f"reward={reward:.4f} terminated={terminated} truncated={truncated}"
        )
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    replay(int(sys.argv[1]), [int(a) for a in sys.argv[2:]])
