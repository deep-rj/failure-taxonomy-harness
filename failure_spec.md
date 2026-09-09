# Spec — Failure Classification for a Multi-Stage Eval Harness

**Status:** frozen before implementation. Any change made *after* code exists must be
recorded in the Changelog at the bottom, with a reason. The point of this file is that
the code does not get to define the semantics.

---

## 1. Problem

**Object level.** A scalar success rate on a multi-stage task is not a diagnosis. Given a
policy that fails, automatically assign each failed episode a failure class describing
*where in the task it broke down*, using only signals the environment exposes — no human
watching video.

## 2. Environment

- `MiniGrid-DoorKey-6x6-v0`, wrapped in `FullyObsWrapper` then `ImgObsWrapper`.
- **Why fully observable:** the default 7x7 egocentric view is partially observable and
  perceptually aliased (turning in place changes the observation without the world
  changing). A memoryless BC policy on aliased inputs fails for perception reasons, which
  would contaminate every failure class. Removing partial observability makes the task
  roughly Markovian so the classes mean what they say. Also, it keeps things simple for the
  scope.
- **Accepted cost:** the memory/exploration dimension is removed entirely. This is a
  scope decision.
- `max_steps`: **50.** Default (250 on 5x5) is ~20x optimal and makes every timeout a
  stall. Derived from the BFS planner's eval-seed length distribution — see §9.

### Verified environment facts

Confirmed empirically on minigrid 3.1.0 / gymnasium 1.3.0. Re-verify if versions differ.

| Fact | Value |
|---|---|
| Action space | `Discrete(7)`: left, right, forward, pickup, drop, toggle, done |
| Success reward | `1 - 0.9 * (step_count / max_steps)` — **never 1.0** |
| Timeout signal | `truncated=True, terminated=False, reward=0` |
| Success signal | `terminated=True, reward > 0` |
| Key pickup | key object **disappears from the grid encoding** — detectable from obs alone |
| Drop | returns key to the grid; `carrying` becomes `None` |
| Toggle on open door | **closes it** (`is_locked=False, is_open=False`) |
| Closed door tile | not walkable, even when unlocked |
| `done` (action 6) | consumes a step, does **not** terminate — a pure budget-burning no-op |
| Goal tile | entering it terminates immediately; no post-goal steps exist |
| Drop target | the tile **in front**, not the agent's own tile; fails if that tile is not free |

## 3. Policy

Behavioral cloning on expert demonstrations. **The policy is a fixture, not a
contribution.** Deliberately weakened into a 40–60% success band via demo count, model
size, or epoch cap.

## 4. Seed policy

Layouts are seed-determined. Demo seeds and eval seeds are **disjoint sets**, fixed in
config and recorded in every output file.

- Demo seeds: `range(0, 150)`
- Eval seeds: `range(10000, 10050)`

**Known seed collisions.** Disjoint seed *numbers* don't guarantee disjoint *layouts* --
`MiniGrid-DoorKey-6x6-v0`'s layout space is small enough that different seeds can
generate the identical grid. Audited via `scripts/planner_audit.py`
(`config_hash`/`layout_hash` in `planner.py`), hashing each seed's actual generated
setup rather than assuming the seed ranges being disjoint is sufficient:

- **Full-setup collisions** (identical room *and* agent start pose -- effectively the
  same episode): **3** eval seeds duplicate a demo seed exactly.
  - Demo seeds: `1, 14, 38`
  - Eval seeds: `10008, 10027, 10044`
  - Pairing: `1↔10008`, `14↔10027`, `38↔10044`
- **Layout-only collisions** (same room/key/door/goal, different agent start pose --
  expected given the small layout space; not itself a leak, tracked for visibility):
  **14** demo layouts reappear somewhere in eval.
  - Demo seeds: `1, 3, 4, 6, 8, 9, 10, 12, 14, 17, 18, 22, 23, 25, 26, 27, 28, 30, 33, 35, 36, 37, 38`
  - Eval seeds: `10001, 10004, 10005, 10007, 10008, 10011, 10012, 10013, 10016, 10017, 10019, 10020, 10022, 10024, 10026, 10027, 10031, 10033, 10036, 10037, 10039, 10040, 10042, 10043, 10044, 10047, 10049`

The full-setup collisions are the actionable ones (they mean the BC policy trains and
is evaluated on the identical episode for those 3 eval seeds). Whether to exclude them
is an open decision -- see §9.

## 5. Failure taxonomy

Applies to failed episodes only. Success is `terminated=True and reward > 0`.

| Class | Name | Meaning |
|---|---|---|
| 1 | `NEVER_PICKED_KEY` | Key was never picked up |
| 2 | `NEVER_UNLOCKED_DOOR` | Key was picked up (possibly dropped/re-picked), door never unlocked |
| 3 | `NEVER_CROSSED_DOOR` | Door was unlocked, agent never crossed through it |
| 4 | `NEVER_REACHED_GOAL` | Agent crossed the door, never reached the goal |

**Precedence:** ascending. The chain is strictly ordered — each stage is a precondition
for the next — so the first unmet milestone determines the class. Classes are mutually
exclusive and, by construction, total over failed episodes.

## 6. Predicates

All milestones are recorded as **first-occurrence step indices** (nullable int), never as
end-of-episode booleans. Terminal-state reads are wrong: drop returns the key to the
grid, and re-toggle closes an opened door, so a snapshot at termination can report a
milestone was never reached when it was.

| Predicate | Definition |
|---|---|
| `key_picked_step` | First step at which the agent is carrying the key |
| `door_unlocked_step` | First step at which the door's `is_locked` becomes `False` (irreversible) |
| `crossed_step` | First step at which the agent, having occupied the door tile, takes a further step in the same direction it entered from |
| `goal_step` | Step at which `terminated=True` with `reward > 0` |

Booleans are derived from these, not stored independently.

**Predicates key on state transitions, never on actions issued.** `pickup`, `drop`, and
`toggle` are all no-ops when the target tile does not permit them, and `done` is a no-op
always. `action == 3` does not mean a pickup occurred; `carrying` transitioning
`None -> key` does. Scaffolding predicates off the action column is the obvious mistake
and produces phantom milestones.

**Note on `door_unlocked_step`:** unlocking is irreversible; *opening* is not. The
predicate keys on the lock bit deliberately.

**Note on `crossed_step`:** occupancy of the door tile is not transit — an agent can step
onto the tile and retreat. The directional definition above is the chosen operationalization.

## 7. Flags

Orthogonal to the class axis. Classes answer *where* it stalled; flags answer *what it
did*. Stored as **counts** (int, ≥0), not booleans — `0` is falsy so any "did this
happen" check still works unchanged, but a count also captures how many times the
policy repeated the behavior.

| Flag | Definition |
|---|---|
| `door_closed_after_unlock` | Count of close events after `door_unlocked_step` and before `crossed_step` (or with no crossing at all) |
| `key_dropped` | Count of successful drop events after `key_picked_step` |

`door_closed_after_unlock` captures the self-sabotage case: the policy undid its own
progress, possibly repeatedly.

## 8. Schemas

### Per-step trajectory — JSONL, one file per episode & one line per step

```
seed, step, action, action_name, agent_pos, agent_dir, agent_dir_name,
carrying, door_locked, door_open, reward, terminated, truncated
```

`action_name`/`agent_dir_name` are human-readable duplicates of the numeric
`action`/`agent_dir` codes (§2's action list; MiniGrid's `DIR_TO_VEC` order for
direction), for manual inspection only. The numeric codes are retained and
are what predicates/classifier key on — §6 already requires keying on state
transitions, never on the action column, so this addition changes no
semantics.

Raw observation arrays are **optional**, behind a flag, for manual spot-checks only. The
classifier must not require them.

**Step indexing:** no row is recorded for the initial post-`reset()` state. The first
recorded row (`step 1`) is the state after the first action. `reset()` does not consume
step budget, consistent with §2's `step_count` semantics — one row per `env.step()` call,
full stop.

### Per-episode record — CSV, one row per episode

```
seed, success, total_steps, terminated, truncated, final_reward, failure_class,
key_picked_step, door_unlocked_step, crossed_step, goal_step,
door_closed_after_unlock, key_dropped
```

## 9. Open items

- [x] `max_steps` value — derived from the BFS planner (`src/failure_taxonomy_harness/planner.py`,
      run via `scripts/planner_audit.py`). All 40 demo seeds and 50 eval seeds were
      planned and their optimal action sequences replay-verified (`verify()`): 90/90
      terminate with `reward > 0` within their planned length, confirming the planner is
      sound. Eval-seed (10000-10049) optimal-length distribution: min=9, median=12,
      mean=12.5, max=18. Chosen `max_steps = 50` (~2.8x the eval max of 18), within the
      2-3x band this item specified.
- [ ] Full-setup demo/eval seed collisions (§4) — decide whether to exclude the 3
      colliding seed pairs (demo `1, 14, 38` ↔ eval `10008, 10027, 10044`) from their
      respective sets, or accept them as-is and note the caveat wherever eval results
      are reported.

## 10. Success criteria

1. **Total coverage** — no episode classified `UNCLASSIFIED`. A hole is a finding, not a
   failure, provided it is named.
2. **Mutual exclusivity** — precedence resolves every case deterministically; zero
   double-counted episodes.
3. **Determinism** — same seed list, two runs, identical per-episode class assignments.
4. **Manual agreement** — ~10 failed episodes hand-inspected against their auto-assigned
   class. This is the ground-truth spot-check and it is not optional. An auto-classifier
   never validated against manual inspection is the same error as treating judge output
   as ground truth.

## 11. Verification plan

- Unit tests over **hand-constructed synthetic trajectories**, one per class, plus:
  - drop case: picked → dropped → terminal state shows no key
  - double-toggle case: unlocked → closed → never crossed
  - step-on-then-retreat case: occupied door tile, did not cross
- no-op spam case: trajectory full of `pickup`/`drop`/`toggle`/`done` actions with no
    state change — all milestones must remain `None`
- Invariant assertion: `goal_step` is not None implies `crossed_step` is not None, and
  milestone indices are strictly increasing where present.
- Determinism check: rerun eval seeds, assert identical assignments.
- Coverage assertion: fail loudly on `UNCLASSIFIED`.
- Coordinate-convention test: assert a known object sits at a known `[x, y]` index for a
  fixed seed, pinning the indexing convention permanently.
- Manual inspection path: render or dump the step sequence for a given seed.

---

## Changelog

| Date | Change | Reason |
|---|---|---|
| — | Initial freeze | — |
| 2026-08-28 | `max_steps` set to 50 (§2, §9) | Closes the §9 open item: derived from the BFS planner's eval-seed optimal-length distribution (max=18) per the 2-3x rule this spec already specified; 350/350 demo+eval seeds replay-verified sound. |
| 2026-08-30 | Demo seed range reduced from `range(0, 300)` to `range(0, 40)` (§4) | User-directed scope reduction for the demo set; not derived from a technical constraint recorded here. |
| 2026-08-30 | Documented demo/eval seed-collision audit (§4) and added an open item for it (§9) | `scripts/planner_audit.py` hashes each seed's actual generated setup (`config_hash`/`layout_hash`) to check demo/eval disjointness holds at the layout level, not just the seed-number level. Found 3 full-setup collisions and 14 layout-only collisions — recorded so any decision to exclude/keep the colliding seeds is deliberate rather than accidental. |
| 2026-08-30 | Documented step-indexing convention (§8): no row for the initial post-`reset()` state, first recorded row is the state after the first action | §8 was silent on the starting index; decided while hand-authoring predicate test fixtures. Needed pinning now so the recorder, predicate tests, and the eventual predicate implementation agree on what "step 1" means. |
| 2026-08-30 | Flags (§7) changed from bool to int count | Discovered while hand-authoring predicate test fixtures: a count captures how many times the policy repeated a self-sabotage behavior (e.g. closing the door twice), which a bool discards. Backward compatible — `bool` is a subtype of `int` in Python, so existing "did this happen" truthiness checks are unaffected. |
| 2026-09-08 | Demo seed range increased from `range(0, 40)` to `range(0, 150)` (§4) | 40 seeds overfit (train/val accuracy gap, ~8% rollout success despite 80%+ step accuracy). Raised incrementally to 150, landing in the §3 40-60% band (52% blended; 47.2% on the 36 eval seeds not colliding with a demo seed per §4/§9). Failure classes stay `NEVER_PICKED_KEY`/`NEVER_UNLOCKED_DOOR` only, manually spot-checked against §5. |
| 2026-09-08 | Per-step trajectory schema (§8) gains `action_name`/`agent_dir_name` alongside the existing numeric `action`/`agent_dir` | Requested for manual debugging (`scripts/record_eval.py`'s per-episode JSONL dumps) — reading `PICKUP`/`DOWN` beats decoding integer codes by hand. Purely additive: the numeric codes are unchanged and remain what predicates/classifier key on (§6). |
| 2026-09-08 | `scripts/record_eval.py` now prints a §5 failure-class distribution after each run, denominator fixed as **failed episodes**, not all episodes | Closes an open aggregation-denominator question flagged during implementation planning ("both are defensible, they differ, and nobody will notice a wrong denominator"). Classes are only ever assigned to failed episodes (§5); a share of *all* episodes would fold "succeeded, no class applies" and "failed at a different stage" into one denominator. Percent of failures answers the question the taxonomy exists to answer. All four classes are always listed, including at count 0, so an unproduced class stays visible instead of disappearing. |