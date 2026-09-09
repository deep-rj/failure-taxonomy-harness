"""BFS expert planner for MiniGrid-DoorKey.

Plans over the analytic transition model in the repo's design discussion, not by
stepping the real env. State: (agent_pos, agent_dir, has_key, door_open). See
failure_spec.md for why this is the frozen abstraction.
"""

import hashlib
import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import gymnasium as gym
from minigrid.core.actions import Actions
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

# (x, y, agent_dir, has_key, door_open)
State = tuple[int, int, int, bool, bool]

# Active search actions only -- see _successors for why drop/done are excluded.
_LEFT, _RIGHT, _FORWARD, _PICKUP, _TOGGLE = (
    int(Actions.left),
    int(Actions.right),
    int(Actions.forward),
    int(Actions.pickup),
    int(Actions.toggle),
)


@dataclass(frozen=True)
class _Layout:
    walls: frozenset[tuple[int, int]]
    key_pos: tuple[int, int]
    door_pos: tuple[int, int]
    goal_pos: tuple[int, int]
    start_pos: tuple[int, int]
    start_dir: int


def _extract_layout(env: gym.Env) -> _Layout:
    """Read the static layout off the grid once. No stepping, no deepcopy."""
    unwrapped = cast(MiniGridEnv, env.unwrapped)
    grid = unwrapped.grid
    walls: set[tuple[int, int]] = set()
    key_pos = door_pos = goal_pos = None

    for i, cell in enumerate(grid.grid):
        if cell is None:
            continue
        # minigrid stores the grid row-major: index = y * width + x.
        x, y = i % grid.width, i // grid.width
        if isinstance(cell, Wall):
            walls.add((x, y))
        elif isinstance(cell, Key):
            key_pos = (x, y)
        elif isinstance(cell, Door):
            door_pos = (x, y)
        elif isinstance(cell, Goal):
            goal_pos = (x, y)

    assert key_pos and door_pos and goal_pos, (  # noqa: S101 -- internal invariant, not input validation
        "DoorKey layout must have a key, door, and goal"
    )

    agent_x, agent_y = (int(v) for v in unwrapped.agent_pos)
    return _Layout(
        walls=frozenset(walls),
        key_pos=key_pos,
        door_pos=door_pos,
        goal_pos=goal_pos,
        start_pos=(agent_x, agent_y),
        start_dir=int(unwrapped.agent_dir),
    )


def _cell_walkable(pos: tuple[int, int], has_key: bool, door_open: bool, layout: _Layout) -> bool:
    if pos in layout.walls:
        return False
    if pos == layout.key_pos:
        return has_key  # key tile is only free once it's been picked up
    if pos == layout.door_pos:
        return door_open
    return True  # floor, goal, or any other static-walkable tile


def _successors(state: State, layout: _Layout) -> Iterable[tuple[State, int]]:
    x, y, direction, has_key, door_open = state
    dx, dy = (int(v) for v in DIR_TO_VEC[direction])
    front = (x + dx, y + dy)

    yield (x, y, (direction - 1) % 4, has_key, door_open), _LEFT
    yield (x, y, (direction + 1) % 4, has_key, door_open), _RIGHT

    if _cell_walkable(front, has_key, door_open, layout):
        yield (front[0], front[1], direction, has_key, door_open), _FORWARD

    if not has_key and front == layout.key_pos:
        yield (x, y, direction, True, door_open), _PICKUP

    if front == layout.door_pos:
        if door_open:
            yield (x, y, direction, has_key, False), _TOGGLE
        elif has_key:
            # DoorKey's single locked door: unlock and open happen atomically.
            # Requiring has_key here (rather than tracking is_locked separately)
            # is sound for shortest paths -- see the transition-model writeup.
            yield (x, y, direction, has_key, True), _TOGGLE

    # `drop` (4) is excluded: the frozen state has no slot for a dropped key's
    # position, and dropping can never shorten a path in DoorKey anyway.
    # `done` (6) is excluded: confirmed no-op that never terminates.


def _bfs(layout: _Layout) -> list[int]:
    start: State = (*layout.start_pos, layout.start_dir, False, False)
    if start[:2] == layout.goal_pos:
        return []

    visited: set[State] = {start}
    queue: deque[State] = deque([start])
    came_from: dict[State, tuple[State, int]] = {}

    while queue:
        state = queue.popleft()
        for next_state, action in _successors(state, layout):
            if next_state in visited:
                continue
            visited.add(next_state)
            came_from[next_state] = (state, action)
            if next_state[:2] == layout.goal_pos:
                return _reconstruct(came_from, next_state)
            queue.append(next_state)

    raise RuntimeError("BFS planner found no solution -- layout has no valid path to goal")


def _reconstruct(came_from: dict[State, tuple[State, int]], goal_state: State) -> list[int]:
    actions: list[int] = []
    state = goal_state
    while state in came_from:
        state, action = came_from[state]
        actions.append(action)
    actions.reverse()
    return actions


def _layout_for_seed(seed: int, env_id: str) -> _Layout:
    env = gym.make(env_id)
    try:
        env.reset(seed=seed)
        return _extract_layout(env)
    finally:
        env.close()


def _layout_payload(layout: _Layout, *, include_start: bool) -> dict[str, object]:
    """Plain-int, JSON-safe view of a layout, for hashing.

    Explicit int() coercion here, not trusting _Layout's field types, so the
    fingerprint can't silently change if a future edit lets a numpy scalar leak
    through -- numpy's scalar repr isn't stable across versions (numpy 2.0 changed
    np.int64's repr from "1" to "np.int64(1)").
    """
    payload: dict[str, object] = {
        "walls": sorted([int(x), int(y)] for x, y in layout.walls),
        "key_pos": [int(v) for v in layout.key_pos],
        "door_pos": [int(v) for v in layout.door_pos],
        "goal_pos": [int(v) for v in layout.goal_pos],
    }
    if include_start:
        payload["start_pos"] = [int(v) for v in layout.start_pos]
        payload["start_dir"] = int(layout.start_dir)
    return payload


def _fingerprint(payload: dict[str, object]) -> str:
    """SHA-256 hex digest of `payload`.

    json.dumps(sort_keys=True) over a plain dict rather than repr() of a tuple:
    keyed by field name (immune to a field-order change in _Layout), and only
    ever fed plain int/str/list (immune to numpy repr differences -- see
    _layout_payload). hashlib, not builtin hash(), for a digest that's stable
    across interpreter runs, not just within one.
    """
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def plan(seed: int, env_id: str = "MiniGrid-DoorKey-6x6-v0") -> list[int]:
    """Shortest action sequence solving `env_id` at `seed`, as MiniGrid action ints."""
    return _bfs(_layout_for_seed(seed, env_id))


def config_hash(seed: int, env_id: str = "MiniGrid-DoorKey-6x6-v0") -> str:
    """Hash of a seed's full setup: static layout + agent start pose.

    Two seeds sharing this hash produce an identical episode -- same room, same
    key/door/goal, same agent starting position and direction.
    """
    layout = _layout_for_seed(seed, env_id)
    return _fingerprint(_layout_payload(layout, include_start=True))


def layout_hash(seed: int, env_id: str = "MiniGrid-DoorKey-6x6-v0") -> str:
    """Hash of a seed's static grid layout only: walls, key, door, goal.

    Excludes the agent's start position/direction (unlike config_hash()). Two
    seeds sharing this hash have the same room/key/door/goal but may place the
    agent differently -- a weaker, more-likely-to-collide notion of "same setup".
    """
    layout = _layout_for_seed(seed, env_id)
    return _fingerprint(_layout_payload(layout, include_start=False))


def verify(seed: int, plan: list[int], env_id: str = "MiniGrid-DoorKey-6x6-v0"):
    env = gym.make(env_id, max_steps=len(plan))
    env.reset(seed=seed)
    terminated = False
    reward = 0.0
    for action in plan:
        _obs, reward, terminated, truncated, _info = env.step(action)
        reward = float(reward)
        if terminated or truncated:
            break
    return terminated and reward > 0


def optimal_lengths(
    seeds: Iterable[int], env_id: str = "MiniGrid-DoorKey-6x6-v0"
) -> dict[int, int]:
    """Optimal solve length per seed, e.g. for calibrating max_steps."""
    return {seed: len(plan(seed, env_id)) for seed in seeds}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the BFS-optimal action sequence for a seed."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-6x6-v0")
    args = parser.parse_args()

    actions = plan(args.seed, args.env_id)
    names = [Actions(a).name for a in actions]
    print(f"seed={args.seed}  length={len(actions)}")
    print(f"actions={actions}")
    print(f"names  ={names}")
