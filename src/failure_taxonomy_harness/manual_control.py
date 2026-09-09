from typing import cast

import gymnasium as gym
import numpy as np
import pygame
from minigrid.core.actions import Actions
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgPartialObsWrapper

# minigrid's own idx->name maps, vectorized so they apply elementwise to a
# whole observation channel at once.
_object_idx_to_name = np.vectorize(IDX_TO_OBJECT.get)
_color_idx_to_name = np.vectorize(IDX_TO_COLOR.get)


def render_obs_grids(image: np.ndarray) -> tuple[str, str]:
    """Split a MiniGrid (width, height, 3) image observation into its object
    and colour grids, using minigrid's own IDX_TO_OBJECT/IDX_TO_COLOR names.

    Channel 0 is the OBJECT_TO_IDX type, channel 1 is COLOR_TO_IDX (channel 2,
    state, is left out here). Both are transposed to [height, width] so each
    grid reads top-to-bottom. Works for either observation mode:
    - Default partial (agent-view) observation: there is no "agent" entry in
      the array at all. Its position is implicit, given only by the fixed
      geometry of gen_obs_grid (always the bottom-centre cell, facing "up"),
      not by anything encoded in the data. That cell just shows what the
      agent is carrying (or "empty").
    - FullyObsWrapper observation: the agent's actual grid cell is labelled
      "agent" (color "red") at its real position, directly in the data.
    """

    def grid_str(names: np.ndarray) -> str:
        return "\n".join(" ".join(f"{cell:>6}" for cell in row) for row in names.T)

    objects = _object_idx_to_name(image[..., 0])
    colors = _color_idx_to_name(image[..., 1])
    return grid_str(objects), grid_str(colors)


class ObservingManualControl(ManualControl):
    """ManualControl that also prints the observation returned by each step."""

    def __init__(self, env, seed=None, print_obs: bool = False) -> None:
        super().__init__(env, seed=seed)
        self.print_obs = print_obs

    def start(self):
        """Same event loop as ManualControl.start(), except it actually sets
        self.closed on a QUIT event. Upstream never does, so the base
        implementation spins once more after env.close() and crashes with
        "pygame.error: video system not initialized" as soon as the window
        is closed.
        """
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    self.closed = True
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def key_handler(self, event):
        """Same as ManualControl.key_handler(), except pressing escape also
        sets self.closed — upstream calls env.close() here too without
        setting it, which crashes start()'s loop the same way as the QUIT
        case above.
        """
        if event.key == "escape":
            self.env.close()
            self.closed = True
            return
        super().key_handler(event)

    def step(self, action: Actions):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        step_count = cast(MiniGridEnv, self.env.unwrapped).step_count
        print(f"step={step_count}, reward={reward:.2f}")
        if self.print_obs:
            if isinstance(obs, dict):
                print(f"obs.direction={obs['direction']}, obs.mission={obs['mission']!r}")
                object_grid, color_grid = render_obs_grids(obs["image"])
                print(f"obs.image objects=\n{object_grid}")
                print(f"obs.image colors=\n{color_grid}")
            else:
                print(f"obs=\n{obs}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-DoorKey-6x6-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument("--tile-size", type=int, help="size at which to render tiles", default=32)
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--full-obs",
        action="store_true",
        help="use FullyObsWrapper for a fully observable grid instead of the agent's partial view",
    )
    parser.add_argument(
        "--print-obs",
        action="store_true",
        help="print the observation (object/colour grids) on each step",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )

    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="human",
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )

    if args.full_obs:
        print("Using full observation")
        env = FullyObsWrapper(env)

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)

    manual_control = ObservingManualControl(env, seed=args.seed, print_obs=args.print_obs)
    manual_control.start()
