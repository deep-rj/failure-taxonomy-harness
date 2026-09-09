"""Canonical demo/eval seed sets (failure_spec.md §4).

Single source of truth for `DEMO_SEEDS` and `EVAL_SEEDS` -- import from here
rather than redefining, so every script that touches the seed sets (demo
generation, training, the planner audit, eval) agrees with the others by
construction instead of by manual upkeep.
"""

# failure_spec.md §4.
DEMO_SEEDS = range(0, 150)
EVAL_SEEDS = range(10000, 10050)
