from typing import Any

from minigrid.core.constants import DIR_TO_VEC


def _get_door_and_cross_positions(step: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    x, y = step["agent_pos"]
    dx, dy = (int(v) for v in DIR_TO_VEC[step["agent_dir"]])
    # Door cross position is simply the tile corresponding to taking two steps
    # in the same direction as the door
    return (x + dx, y + dy), (x + 2 * dx, y + 2 * dy)


def _is_door_crossed(
    previous_step: dict,
    step: dict,
    door_pos: tuple[int, int] | None,
    door_cross_pos: tuple[int, int] | None,
) -> bool:
    if not previous_step:
        return False
    return previous_step["agent_pos"] == door_pos and step["agent_pos"] == door_cross_pos


def compute_predicates(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """First-occurrence milestone step indices for a single episode's trajectory.

    `steps` is a list of per-step dicts using the §8 per-step field names
    (seed, step, action, agent_pos, agent_dir, carrying, door_locked,
    door_open, reward, terminated, truncated), ordered by step, starting at
    step 1 (no row for the initial post-reset() state — see §8).

    Returns a dict with the §6 milestones (key_picked_step, door_unlocked_step,
    crossed_step, goal_step -- each a step index or None) and the §7 flags
    (door_closed_after_unlock, key_dropped -- each an int count, >= 0).
    """
    door_pos: tuple[int, int] | None = None
    door_cross_pos: tuple[int, int] | None = None
    key_picked: int | None = None
    door_unlocked: int | None = None
    crossed: int | None = None
    goal: int | None = None
    door_closed_after_unlock: int = 0
    key_dropped: int = 0
    previous_step: dict[str, Any] = {}

    for step in steps:
        step_number = step.get("step")

        if goal is None and step.get("terminated") and step.get("reward", 0) > 0:
            goal = step_number

        if crossed is None and _is_door_crossed(previous_step, step, door_pos, door_cross_pos):
            crossed = step_number

        if door_unlocked is None and not step.get("door_locked"):
            door_unlocked = step_number
            if door_pos is None:
                door_pos, door_cross_pos = _get_door_and_cross_positions(step)

        if key_picked is None and step.get("carrying") and not previous_step.get("carrying"):
            key_picked = step_number

        if crossed is None and not step.get("door_open") and previous_step.get("door_open"):
            door_closed_after_unlock += 1

        if not step.get("carrying") and previous_step.get("carrying"):
            key_dropped += 1

        previous_step = step

    return {
        "key_picked_step": key_picked,
        "door_unlocked_step": door_unlocked,
        "crossed_step": crossed,
        "goal_step": goal,
        "door_closed_after_unlock": door_closed_after_unlock,
        "key_dropped": key_dropped,
    }
