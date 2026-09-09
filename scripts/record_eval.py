"""Record per-step trajectories and per-episode failure classifications for a
BC policy rollout sweep -- the debug/inspection path for "where did this
seed's episode go wrong" (failure_spec.md §8).

For each seed, runs eval.rollout() and writes:
  - one §8 per-step JSONL trajectory file per episode
    (<out-dir>/trajectories/seed_{seed}.jsonl)
  - one §8 per-episode CSV summary row, all episodes in a single file
    (<out-dir>/episodes.csv)

Failed episodes' trajectories are run through predicates.compute_predicates()
and classifier.classify() to fill in the milestone/flag/failure_class columns;
classify() is never called on a successful episode (its own precondition --
see classifier.py), so those columns are left blank for successes.

Usage:
    uv run python scripts/record_eval.py
    uv run python scripts/record_eval.py --seeds 10000 10001 --out-dir /tmp/eval
"""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from failure_taxonomy_harness.bc import DoorKeyPolicyModel
from failure_taxonomy_harness.classifier import FailureClass, classify
from failure_taxonomy_harness.env import DEFAULT_ENV_ID, DEFAULT_MAX_STEPS
from failure_taxonomy_harness.eval import DEFAULT_MODEL_PATH, EVAL_SEEDS, rollout
from failure_taxonomy_harness.predicates import compute_predicates

DEFAULT_OUT_DIR = Path("data/eval")

# §8 per-episode CSV column order.
EPISODE_FIELDS = [
    "seed",
    "success",
    "total_steps",
    "terminated",
    "truncated",
    "final_reward",
    "failure_class",
    "key_picked_step",
    "door_unlocked_step",
    "crossed_step",
    "goal_step",
    "door_closed_after_unlock",
    "key_dropped",
]


def write_trajectory(seed: int, steps: list[dict[str, Any]], out_dir: Path) -> Path:
    out_path = out_dir / "trajectories" / f"seed_{seed}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for step in steps:
            f.write(json.dumps(step) + "\n")
    return out_path


def episode_record(seed: int, result: dict[str, Any]) -> dict[str, Any]:
    """Builds one §8 per-episode CSV row from one seed's `rollout()` result.

    `failure_class` (and the milestone/flag columns that feed it) are only
    meaningful for failed episodes -- classify()'s precondition forbids
    calling it on a success, so successes get milestones computed (harmless,
    and useful for spot-checking) but `failure_class` left as None.
    """
    steps = result["steps"]
    last = steps[-1]
    success = result["success"]
    milestones = compute_predicates(steps)

    failure_class = None
    if not success:
        outcome = {
            "success": success,
            "terminated": last["terminated"],
            "truncated": last["truncated"],
        }
        failure_class = classify(milestones, outcome)

    return {
        "seed": seed,
        "success": success,
        "total_steps": len(steps),
        "terminated": last["terminated"],
        "truncated": last["truncated"],
        "final_reward": last["reward"],
        "failure_class": failure_class,
        **milestones,
    }


def failure_class_distribution(records: list[dict[str, Any]]) -> str:
    """Formats the §5 failure-class distribution over a batch of episodes.

    Denominator is failed episodes, not all episodes -- both are defensible,
    so it's easy to pick one implicitly without noticing. Classes are only
    ever assigned to failed episodes (§5) -- computing a class's share of *all* episodes
    would silently fold "the policy succeeded, no class applies" and "the
    policy failed at a different stage" into the same denominator. Percent
    of failures answers the question the taxonomy exists to answer: given
    that this policy failed, where did it break down. All four §5 classes
    are listed even at count 0, so a class the run never produced stays
    visible rather than disappearing from the report.
    """
    failed = [r for r in records if not r["success"]]
    counts = Counter(r["failure_class"] for r in failed)

    lines = [
        f"failure_class_distribution ({len(failed)} failed / {len(records)} total, "
        "% of failed episodes):"
    ]
    for cls in FailureClass:
        if cls is FailureClass.UNCLASSIFIED:
            continue
        n = counts.get(cls, 0)
        pct = f"{n / len(failed):.1%}" if failed else "n/a"
        lines.append(f"  {cls.value:22s} {n:3d}  ({pct})")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(EVAL_SEEDS))
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    model = DoorKeyPolicyModel()
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "episodes.csv"

    successes = 0
    records = []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        for seed in args.seeds:
            result = rollout(model, seed, args.env_id, args.max_steps)
            write_trajectory(seed, result["steps"], args.out_dir)
            record = episode_record(seed, result)
            writer.writerow(record)
            records.append(record)
            successes += record["success"]
            print(
                f"seed={seed:5d}  success={record['success']!s:5}  "
                f"failure_class={record['failure_class']}"
            )

    total = len(args.seeds)
    print(f"\nwrote {total} episodes to {args.out_dir} ({csv_path.name} + trajectories/)")
    print(f"success_rate={successes / total:.2%} ({successes}/{total})")
    print(failure_class_distribution(records))


if __name__ == "__main__":
    main()
