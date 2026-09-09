"""Shared helpers for hand-authoring trajectory fixtures.

Kept deliberately dumb: `step()` only fills in the fields that are constant
across most rows of a hand-written trajectory (seed, reward, terminated,
truncated, and the pre-unlock/pre-pickup defaults for carrying/door state).
It does not track or carry forward state between calls -- every row's actual
values are still whatever you pass in, so the trajectory stays hand-verifiable
line by line.

Default seed is 31: a real MiniGrid-DoorKey-6x6-v0 seed (not an assumed
layout), chosen so every fixture in tests/test_predicates.py is replay-checkable
against the actual env, per `scripts/replay_seed.py`. Its layout: key at (1, 2),
door at (3, 2) (the only gap in the wall splitting the room at x=3), goal at
(4, 4), agent starts at (2, 2) facing east (dir 0).
"""

# Action space (§2) -- Discrete(7) in this fixed order.
LEFT = 0
RIGHT = 1
FORWARD = 2
PICKUP = 3
DROP = 4
TOGGLE = 5
DONE = 6


def step(
    step: int,
    action: int,
    pos: tuple[int, int],
    dir: int,
    *,
    seed: int = 31,
    carrying: bool = False,
    door_locked: bool = True,
    door_open: bool = False,
    reward: float = 0,
    terminated: bool = False,
    truncated: bool = False,
) -> dict:
    """Build one §8-shaped per-step row.

    Positional args are the fields that change on essentially every row;
    keyword args default to the pre-pickup/pre-unlock starting state and are
    only worth passing when they differ from the previous row.
    """
    return {
        "seed": seed,
        "step": step,
        "action": action,
        "agent_pos": pos,
        "agent_dir": dir,
        "carrying": carrying,
        "door_locked": door_locked,
        "door_open": door_open,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }


def milestones(
    *,
    key_picked: int | None = None,
    door_unlocked: int | None = None,
    crossed: int | None = None,
    goal: int | None = None,
    door_closed_after_unlock: int = 0,
    key_dropped: int = 0,
) -> dict:
    """Build the expected `compute_predicates()` return value.

    All keyword-only, all default to the "nothing happened" state (every
    milestone None, every flag count 0), since which milestone(s)/flag(s) a
    given test cares about varies per case -- pass only the ones that should
    be set. Flags are counts (§7), not bools -- pass the number of occurrences.

    Doubles as the milestones+flags half of `classify()`'s input in
    test_classifier.py -- see `outcome()` below for the other half.
    """
    return {
        "key_picked_step": key_picked,
        "door_unlocked_step": door_unlocked,
        "crossed_step": crossed,
        "goal_step": goal,
        "door_closed_after_unlock": door_closed_after_unlock,
        "key_dropped": key_dropped,
    }


def outcome(
    *,
    success: bool = False,
    terminated: bool = False,
    truncated: bool = False,
) -> dict:
    """Build the minimal per-episode outcome bundle for `classify()`'s
    "applies to failed episodes only" precondition (§5).

    Deliberately narrow: just success/terminated/truncated, not the full §8
    per-episode record -- total_steps and final_reward play no role in
    classification and don't belong in a classifier test fixture.

    All keyword-only, all default to False; pass only the ones a given test
    cares about.
    """
    return {
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
    }
