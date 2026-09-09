from enum import StrEnum


class FailureClass(StrEnum):
    """Failure class assigned to a failed episode (§5), in the spec's
    ascending precedence order -- the chain each episode is checked against.

    A `StrEnum` rather than a plain `Enum`: members compare equal to and
    serialize as their §5 name (`FailureClass.NEVER_PICKED_KEY ==
    "NEVER_PICKED_KEY"`), so the §8 per-episode CSV's `failure_class` column
    needs no extra `.value` conversion at write time.

    `UNCLASSIFIED` is not a valid `classify()` return value -- it exists only
    as the sentinel the §10 coverage test asserts is never produced.
    """

    NEVER_PICKED_KEY = "NEVER_PICKED_KEY"
    NEVER_UNLOCKED_DOOR = "NEVER_UNLOCKED_DOOR"
    NEVER_CROSSED_DOOR = "NEVER_CROSSED_DOOR"
    NEVER_REACHED_GOAL = "NEVER_REACHED_GOAL"
    UNCLASSIFIED = "UNCLASSIFIED"


def classify(milestones: dict, outcome: dict) -> FailureClass:
    """Assign a failure class (§5) to one failed episode.

    `milestones` is a §6 milestones + §7 flags dict, shaped like
    `compute_predicates()`'s return value (key_picked_step, door_unlocked_step,
    crossed_step, goal_step, door_closed_after_unlock, key_dropped).

    `outcome` carries just enough of the §8 per-episode record to guard this
    function's precondition: {"success": bool, "terminated": bool,
    "truncated": bool}. Applies to failed episodes only (§5) -- callers must
    not invoke this on a successful episode.

    Precedence is ascending (§5): the first unmet milestone in the chain
    key_picked_step -> door_unlocked_step -> crossed_step -> goal_step
    determines the class. Must never return `FailureClass.UNCLASSIFIED` for a
    failed episode (§10).
    """
    if outcome["success"]:
        return FailureClass.UNCLASSIFIED

    if milestones["key_picked_step"] is None:
        return FailureClass.NEVER_PICKED_KEY
    elif milestones["door_unlocked_step"] is None:
        return FailureClass.NEVER_UNLOCKED_DOOR
    elif milestones["crossed_step"] is None:
        return FailureClass.NEVER_CROSSED_DOOR
    elif milestones["goal_step"] is None:
        return FailureClass.NEVER_REACHED_GOAL

    return FailureClass.UNCLASSIFIED
