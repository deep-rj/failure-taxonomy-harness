"""Shared MiniGrid env construction (failure_spec.md §2).

Wraps the base env FullyObsWrapper -> ImgObsWrapper for full observability --
see failure_spec.md §2 for why (perceptual aliasing under the default
partially-observable view).
"""

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

DEFAULT_ENV_ID = "MiniGrid-DoorKey-6x6-v0"
DEFAULT_MAX_STEPS = 50  # failure_spec.md §2

# failure_spec.md §2 -- Discrete(7) action space, fixed order.
ACTION_NAMES = ("LEFT", "RIGHT", "FORWARD", "PICKUP", "DROP", "TOGGLE", "DONE")

# Index = agent_dir; order matches minigrid.core.constants.DIR_TO_VEC
# ((1,0), (0,1), (-1,0), (0,-1)) -- +x is right, +y is down.
DIR_NAMES = ("RIGHT", "DOWN", "LEFT", "UP")


def make_env(env_id: str = DEFAULT_ENV_ID, max_steps: int = DEFAULT_MAX_STEPS) -> gym.Env:
    env = gym.make(env_id, max_steps=max_steps)
    env = FullyObsWrapper(env)
    return ImgObsWrapper(env)
