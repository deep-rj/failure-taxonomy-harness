"""Audit the BFS planner's output over the demo and eval seed ranges.

Runs plan() + verify() over the demo and eval seed ranges fixed in failure_spec.md
section 4 to confirm the planner is sound, derives max_steps candidates from the
eval-seed optimal-length distribution per the open item in section 9, and hashes
each seed's setup to check for demo/eval collisions (same episode, or just the
same static room layout).
"""

import time
from collections.abc import Callable

from failure_taxonomy_harness.planner import config_hash, layout_hash, plan, verify
from failure_taxonomy_harness.seeds import DEMO_SEEDS, EVAL_SEEDS

# A handful of boundary seeds from both ranges, to fail fast before the full run.
SMOKE_SEEDS = [0, 1, 2, 39, 10000, 10001, 10049]


def _describe(seeds: range) -> str:
    """ "start-end" (inclusive) for a range, e.g. range(0, 40) -> "0-39"."""
    return f"{seeds.start}-{seeds.stop - 1}"


def verify_seeds(label: str, seeds: range | list[int]) -> dict[int, int]:
    """Plan + verify every seed in `seeds`. Returns {seed: optimal_length}."""
    start = time.monotonic()
    lengths: dict[int, int] = {}
    failures: list[int] = []

    for seed in seeds:
        actions = plan(seed)
        if not verify(seed, actions):
            failures.append(seed)
        else:
            lengths[seed] = len(actions)

    elapsed = time.monotonic() - start
    print(f"[{label}] {len(lengths)}/{len(seeds)} verified in {elapsed:.1f}s")

    if failures:
        raise RuntimeError(f"[{label}] verify() failed for seeds: {failures}")

    return lengths


def report_distribution(label: str, lengths: dict[int, int]) -> None:
    values = sorted(lengths.values())
    n = len(values)
    lo, hi = values[0], values[-1]
    mean = sum(values) / n
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2

    print(f"[{label}] optimal length distribution over {n} seeds:")
    print(f"  min={lo}  median={median}  mean={mean:.1f}  max={hi}")
    for factor in (2, 2.5, 3):
        print(f"  max_steps candidate at {factor}x max optimal ({hi}): {round(hi * factor)}")


def report_hash_intersection(
    label: str, hash_fn: Callable[[int], str], demo_seeds: range, eval_seeds: range
) -> None:
    """Hash each seed with `hash_fn` and check for demo/eval overlap.

    Demo and eval seeds are numerically disjoint by construction, but that doesn't
    guarantee minigrid's layout generator can't produce the *same* setup for two
    different seeds. This checks that directly, off the actual generated layouts.
    """
    demo_hashes = {seed: hash_fn(seed) for seed in demo_seeds}
    eval_hashes = {seed: hash_fn(seed) for seed in eval_seeds}
    shared = set(demo_hashes.values()) & set(eval_hashes.values())

    print(f"[{label}] demo: {len(set(demo_hashes.values()))} unique / {len(demo_hashes)} seeds")
    print(f"[{label}] eval: {len(set(eval_hashes.values()))} unique / {len(eval_hashes)} seeds")

    if shared:
        demo_dupes = [seed for seed, h in demo_hashes.items() if h in shared]
        eval_dupes = [seed for seed, h in eval_hashes.items() if h in shared]
        print(f"[{label}] {len(shared)} value(s) shared between demo and eval:")
        print(f"  demo seeds: {demo_dupes}")
        print(f"  eval seeds: {eval_dupes}")
    else:
        print(f"[{label}] intersection: empty -- no shared values between demo and eval")


def main() -> None:
    print("=== smoke test ===")
    verify_seeds("smoke", SMOKE_SEEDS)

    print(f"\n=== demo seeds ({_describe(DEMO_SEEDS)}) ===")
    verify_seeds("demo", DEMO_SEEDS)

    print(f"\n=== eval seeds ({_describe(EVAL_SEEDS)}) ===")
    eval_lengths = verify_seeds("eval", EVAL_SEEDS)

    print()
    report_distribution("eval", eval_lengths)

    print("\n=== config hash intersection (full setup: layout + agent start) ===")
    report_hash_intersection("config", config_hash, DEMO_SEEDS, EVAL_SEEDS)

    print("\n=== layout hash intersection (static layout only, agent start ignored) ===")
    report_hash_intersection("layout", layout_hash, DEMO_SEEDS, EVAL_SEEDS)


if __name__ == "__main__":
    main()
