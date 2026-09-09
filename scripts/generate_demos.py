"""Generate BC training demonstrations for the demo seed set.

For each demo seed (failure_taxonomy_harness.seeds.DEMO_SEEDS, per
failure_spec.md §4), plans the optimal
action sequence via the BFS planner (src/failure_taxonomy_harness/planner.py)
and replays it in the real, wrapped env (FullyObsWrapper -> ImgObsWrapper,
matching §2), recording one (obs, action) pair per step.

Each pair's `obs` is the env state BEFORE `action` is taken -- what the policy
would have seen and acted on -- never the resulting state. This is the
standard behavioral-cloning convention, and deliberately different from the
§8 per-step trajectory schema (which records POST-action state for the
predicate/classifier pipeline downstream of a policy rollout, not for
training data).

Output: one JSONL file per seed, <out-dir>/seed_{seed:04d}.jsonl, one line
per (obs, action) pair:

    {"seed": int, "step": int, "obs": <nested list, image>, "action": int}

`step` is 0-indexed within the episode -- the index of the action about to be
taken from this obs -- unlike §8's 1-indexed post-action step numbering.

Usage:
    uv run python scripts/generate_demos.py
    uv run python scripts/generate_demos.py --seeds 0 1 2 --out-dir /tmp/demos
"""

import json
from pathlib import Path

from failure_taxonomy_harness.env import DEFAULT_ENV_ID, DEFAULT_MAX_STEPS, make_env
from failure_taxonomy_harness.planner import plan
from failure_taxonomy_harness.seeds import DEMO_SEEDS

DEFAULT_OUT_DIR = Path("data/demos")


def generate_demo(
    seed: int,
    *,
    env_id: str = DEFAULT_ENV_ID,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> list[dict]:
    """Plan + replay one seed. Returns the list of (obs, action) records.

    Raises if the planned actions don't terminate the episode successfully --
    a planner/env mismatch is a bug, not a demo to silently emit.
    """
    actions = plan(seed, env_id)
    env = make_env(env_id, max_steps)
    try:
        obs, _info = env.reset(seed=seed)
        records: list[dict] = []
        terminated = False
        reward = 0.0
        for step_idx, action in enumerate(actions):
            records.append(
                {
                    "seed": seed,
                    "step": step_idx,
                    "obs": obs.tolist(),
                    "action": action,
                }
            )
            obs, reward, terminated, _truncated, _info = env.step(action)

        if actions and not (terminated and float(reward) > 0):
            raise RuntimeError(
                f"seed={seed}: planned actions did not succeed "
                f"(terminated={terminated}, reward={reward})"
            )
        return records
    finally:
        env.close()


def write_demo(seed: int, records: list[dict], out_dir: Path) -> Path:
    out_path = out_dir / f"seed_{seed:04d}.jsonl"
    with out_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEMO_SEEDS))
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    for seed in args.seeds:
        records = generate_demo(seed, env_id=args.env_id, max_steps=args.max_steps)
        out_path = write_demo(seed, records, args.out_dir)
        total_steps += len(records)
        print(f"seed={seed:4d}  steps={len(records):3d}  -> {out_path}")

    print(
        f"\n{len(args.seeds)} demos written, {total_steps} total (obs, action) "
        f"pairs, to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
