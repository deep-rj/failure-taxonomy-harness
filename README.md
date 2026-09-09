# failure-taxonomy-harness

Automatic failure classification for a behavioral-cloning policy on
`MiniGrid-DoorKey-6x6-v0`. Given a policy that fails an episode, assigns a
failure class describing *where in the task it broke down*, using only
signals the environment exposes — no human video review.

See [`failure_spec.md`](failure_spec.md) for the frozen problem spec
(environment setup, failure taxonomy, and classification rules).

## Setup

```bash
uv sync
```

## Usage

The pipeline has three stages, each a standalone script:

**1. Generate expert demonstrations.** Plans each demo seed with the BFS planner
(`src/failure_taxonomy_harness/planner.py`) and replays it in the real env, writing
one `(obs, action)` JSONL file per seed to `data/demos/`.

```bash
uv run python scripts/generate_demos.py
```

**2. Train the (deliberately weakened) BC policy.** A single-hidden-layer MLP,
tuned via `--num-epochs` to land in the 40-60% success band that failure_spec.md
§3 specifies — the policy is a fixture for the classifier to exercise, not the
point of the project. Writes a checkpoint to `models/bc_policy.pt`.

```bash
uv run python scripts/train_bc.py
```

**3. Roll out on the eval seeds and classify every failure.** Runs the trained
policy on the eval seeds, records each episode's full trajectory, and assigns
every failed episode a failure class (§5) via `predicates.py` + `classifier.py`.
Writes per-step JSONL trajectories and a per-episode CSV to `data/eval/`.

```bash
uv run python scripts/record_eval.py
```

Example output from a run (exact numbers vary run to run — training isn't seeded,
so success rate is only guaranteed to land in the 40-60% band, not at this exact
value):

```
success_rate=52.00% (26/50)
failure_class_distribution (24 failed / 50 total, % of failed episodes):
  NEVER_PICKED_KEY        20  (83.3%)
  NEVER_UNLOCKED_DOOR      4  (16.7%)
  NEVER_CROSSED_DOOR       0  (0.0%)
  NEVER_REACHED_GOAL       0  (0.0%)
```

## Manual play

`scripts/manual_control.py` drives the env with your own keyboard input instead
of a trained policy — useful for building intuition about a seed's layout or
spot-checking the environment facts §2 documents.

```bash
uv run python scripts/manual_control.py --seed 31
```

Arrow keys turn/move forward; `space` toggles the door; `pageup`/`tab` pick up
and `pagedown`/`left shift` drop; `backspace` resets the episode; `escape`
closes the window and quits. `enter` issues the `done` action, which per §2
is a no-op that does **not** end the episode -- reaching the goal or timing
out does, and both auto-reset to a fresh episode of the same seed.

Add `--print-obs` to also print the observation's object/colour grids and
per-step reward to the terminal, or `--full-obs` to print the
fully-observable grid `env.py` uses (§2) instead of the agent's default
partial view.

## Project structure

- `src/failure_taxonomy_harness/` — env setup, BFS planner, BC model, predicates,
  and the classifier itself (a pure function: trajectory in, failure class out —
  no env calls, no I/O, unit-testable against synthetic trajectories alone).
- `scripts/` — CLI entry points for the three pipeline stages above, plus
  `planner_audit.py` (seed-collision audit) and `replay_seed.py` (manual
  hand-verification against the real env).
- `tests/` — hand-authored predicate/classifier fixtures (§11), each replay-checked
  against a real seed rather than an assumed layout.

## Testing

```bash
uv run pytest
```
