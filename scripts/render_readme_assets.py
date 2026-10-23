"""Render the README's visuals from a recorded eval run (scripts/record_eval.py).

Replays each chosen seed's recorded actions (<eval-dir>/trajectories/) in the
real env and draws it frame by frame, so every GIF shows exactly the episode
its <eval-dir>/episodes.csv row describes -- no model checkpoint needed. Writes
to <out-dir>:
  - doorkey_annotated.png: the success seed's start state, with a legend of
    the agent/key/door/goal sprites cropped from that same frame
  - success.gif, fail_<failure_class>.gif: one rollout per outcome, captioned
    with the step counter, the §6 milestones lighting up as they are first
    reached, and the final outcome / §5 failure class

Default seeds: the first success and the first failure of each §5 class
present in episodes.csv, skipping the §4 full-setup demo/eval collisions.

Usage:
    uv run --group viz python scripts/render_readme_assets.py
    uv run --group viz python scripts/render_readme_assets.py --seeds 10003 10020
"""

import csv
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from PIL import Image, ImageDraw, ImageFont

from failure_taxonomy_harness.classifier import FailureClass
from failure_taxonomy_harness.env import DEFAULT_MAX_STEPS, make_env

DEFAULT_EVAL_DIR = Path("data/eval")
DEFAULT_OUT_DIR = Path("docs/assets")

# failure_spec.md §4 -- eval seeds that replay a demo seed's exact episode.
COLLIDING_EVAL_SEEDS = {10008, 10027, 10044}

MILESTONES = [
    ("key_picked_step", "key"),
    ("door_unlocked_step", "unlock"),
    ("crossed_step", "cross"),
    ("goal_step", "goal"),
]

TILE_SIZE = 48
LEGEND_TILE_SIZE = 64
STEP_MS = 250
FINAL_FRAME_MS = 2500

BG = (31, 35, 40)
TEXT = (230, 237, 243)
MUTED = (125, 133, 144)
GREEN = (63, 185, 80)
RED = (248, 81, 73)

FONT = ImageFont.load_default(size=15)
SMALL_FONT = ImageFont.load_default(size=13)


def _optional_int(value: str) -> int | None:
    return int(value) if value else None


def load_episodes(eval_dir: Path) -> dict[int, dict[str, Any]]:
    with (eval_dir / "episodes.csv").open() as f:
        rows = list(csv.DictReader(f))
    return {
        int(row["seed"]): {
            "success": row["success"] == "True",
            "failure_class": row["failure_class"] or None,
            **{field: _optional_int(row[field]) for field, _ in MILESTONES},
        }
        for row in rows
    }


def default_seeds(episodes: dict[int, dict[str, Any]]) -> list[int]:
    chosen: dict[str | None, int] = {}
    for seed, episode in episodes.items():
        if seed in COLLIDING_EVAL_SEEDS:
            continue
        outcome = None if episode["success"] else episode["failure_class"]
        chosen.setdefault(outcome, seed)
    order = [None, *(cls.value for cls in FailureClass)]
    return [chosen[outcome] for outcome in order if outcome in chosen]


def load_actions(eval_dir: Path, seed: int) -> list[dict[str, Any]]:
    path = eval_dir / "trajectories" / f"seed_{seed}.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f]


def replay_frames(seed: int, steps: list[dict[str, Any]], tile_size: int) -> list[np.ndarray]:
    """One frame for the post-reset() state, then one per recorded step.

    Raises if the replay diverges from the recording, since the GIF would then
    no longer show the episode its CSV row classifies.
    """
    env = make_env(max_steps=DEFAULT_MAX_STEPS)
    try:
        env.reset(seed=seed)
        unwrapped = cast(MiniGridEnv, env.unwrapped)
        frames = [unwrapped.get_frame(highlight=False, tile_size=tile_size)]
        for step in steps:
            env.step(step["action"])
            pos = [int(v) for v in unwrapped.agent_pos]
            if pos != step["agent_pos"] or (unwrapped.carrying is not None) != step["carrying"]:
                raise RuntimeError(f"seed {seed} replay diverged at step {step['step']}")
            frames.append(unwrapped.get_frame(highlight=False, tile_size=tile_size))
        return frames
    finally:
        env.close()


def _first_unmet_milestone(episode: dict[str, Any]) -> str | None:
    for field, _ in MILESTONES:
        if episode[field] is None:
            return field
    return None


def caption_frame(
    frame: np.ndarray, seed: int, step: int, episode: dict[str, Any], final: bool
) -> Image.Image:
    grid = Image.fromarray(frame)
    width = grid.width
    caption_h = 84
    out = Image.new("RGB", (width, grid.height + caption_h), BG)
    out.paste(grid, (0, 0))
    draw = ImageDraw.Draw(out)
    top = grid.height

    draw.text((10, top + 8), f"seed {seed}", font=FONT, fill=TEXT)
    draw.text(
        (width - 10, top + 8),
        f"step {step}/{DEFAULT_MAX_STEPS}",
        font=FONT,
        fill=MUTED,
        anchor="ra",
    )

    failed_at = _first_unmet_milestone(episode) if final and not episode["success"] else None
    slot = width / len(MILESTONES)
    for i, (field, label) in enumerate(MILESTONES):
        cx = int(slot * i + 16)
        cy = top + 41
        reached_at = episode[field]
        box = (cx - 6, cy - 6, cx + 6, cy + 6)
        if reached_at is not None and reached_at <= step:
            draw.ellipse(box, fill=GREEN)
            color = TEXT
        elif field == failed_at:
            draw.ellipse(box, fill=RED)
            color = RED
        else:
            draw.ellipse(box, outline=MUTED, width=2)
            color = MUTED
        draw.text((cx + 11, cy), label, font=SMALL_FONT, fill=color, anchor="lm")

    if final:
        if episode["success"]:
            text, color = "SUCCESS", GREEN
        else:
            text, color = f"FAIL · {episode['failure_class']}", RED
        draw.text((10, top + 60), text, font=FONT, fill=color)
    return out


def render_gif(
    seed: int, episode: dict[str, Any], steps: list[dict[str, Any]], out_path: Path
) -> None:
    frames = replay_frames(seed, steps, TILE_SIZE)
    last = len(frames) - 1
    images = [caption_frame(f, seed, i, episode, final=i == last) for i, f in enumerate(frames)]
    durations = [STEP_MS] * last + [FINAL_FRAME_MS]
    images[0].save(
        out_path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True
    )


def _find(unwrapped: MiniGridEnv, kind: type) -> tuple[int, int]:
    for i, cell in enumerate(unwrapped.grid.grid):
        if isinstance(cell, kind):
            return (i % unwrapped.grid.width, i // unwrapped.grid.width)
    raise ValueError(f"no {kind.__name__} in grid")


def render_annotated(seed: int, out_path: Path) -> None:
    env = make_env(max_steps=DEFAULT_MAX_STEPS)
    try:
        env.reset(seed=seed)
        unwrapped = cast(MiniGridEnv, env.unwrapped)
        grid = Image.fromarray(unwrapped.get_frame(highlight=False, tile_size=LEGEND_TILE_SIZE))
        agent_pos = (int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1]))
        legend = [
            (agent_pos, "agent (triangle = facing)"),
            (_find(unwrapped, Key), "key"),
            (_find(unwrapped, Door), "locked door"),
            (_find(unwrapped, Goal), "goal"),
            (_find(unwrapped, Wall), "wall"),
        ]
    finally:
        env.close()

    icon = 36
    row_h = icon + 12
    out = Image.new("RGB", (grid.width, grid.height + row_h * ((len(legend) + 1) // 2) + 8), BG)
    out.paste(grid, (0, 0))
    draw = ImageDraw.Draw(out)
    col_w = grid.width // 2
    for i, ((x, y), label) in enumerate(legend):
        tile = grid.crop(
            (
                x * LEGEND_TILE_SIZE,
                y * LEGEND_TILE_SIZE,
                (x + 1) * LEGEND_TILE_SIZE,
                (y + 1) * LEGEND_TILE_SIZE,
            )
        ).resize((icon, icon), Image.Resampling.LANCZOS)
        left = (i % 2) * col_w + 12
        top = grid.height + 10 + (i // 2) * row_h
        out.paste(tile, (left, top))
        draw.text(
            (left + icon + 10, top + icon // 2), label, font=SMALL_FONT, fill=TEXT, anchor="lm"
        )
    out.save(out_path, optimize=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    episodes = load_episodes(args.eval_dir)
    seeds = args.seeds if args.seeds else default_seeds(episodes)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    successes = [s for s in seeds if episodes[s]["success"]]
    if successes:
        path = args.out_dir / "doorkey_annotated.png"
        render_annotated(successes[0], path)
        print(f"wrote {path} (seed {successes[0]})")

    for seed in seeds:
        episode = episodes[seed]
        name = "success" if episode["success"] else f"fail_{episode['failure_class'].lower()}"
        path = args.out_dir / f"{name}.gif"
        render_gif(seed, episode, load_actions(args.eval_dir, seed), path)
        print(f"wrote {path} (seed {seed}, {path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
