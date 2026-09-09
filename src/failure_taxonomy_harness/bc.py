import json
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

_DOOR_OBJECT_ID = 4
_AGENT_OBJECT_ID = 10
_OBJECT_ID_MAP = {1: 0, 2: 1, 4: 2, 5: 3, 8: 4, 10: 5}
_COLOR_ID_MAP = {0: 0, 1: 1, 4: 2, 5: 3}


def _create_lookup_tensor(mapper: dict[int, int]) -> Tensor:
    lookup_tensor = torch.zeros(max(mapper.keys()) + 1, dtype=torch.long)
    mapper_keys = torch.tensor(list(mapper.keys()), dtype=torch.long)
    mapper_values = torch.tensor(list(mapper.values()), dtype=torch.long)
    lookup_tensor[mapper_keys] = mapper_values
    return lookup_tensor


_OBJECT_LOOKUP = _create_lookup_tensor(_OBJECT_ID_MAP)
_COLOR_LOOKUP = _create_lookup_tensor(_COLOR_ID_MAP)


def encode_obs(observation: list[list[list[int]]]) -> Tensor:
    """Encodes observations of the form (n, n, 3) into one-hot encoded
    features of the form (n, n, 17).

    The 17 features correspond to:
    6 objects, 4 colors, 3 door states, 4 agent directions

    Finally, the features are flattened into a 1-D vector of shape n * n * 17
    and converted to `torch.float32`.

    Note: For door states and agent directions, the encoded features are
    zeroed out for all cells that are neither door not agent respectively.
    """
    obs_tensor = torch.tensor(observation, dtype=torch.long)
    object_ids = obs_tensor[..., 0]
    color_ids = obs_tensor[..., 1]
    states = obs_tensor[..., 2]

    # Extract states only for door and agent
    # For door, 0 - open, 1 - closed, 2 - locked
    # For agent, it shows direction, 0 - right, 1 - down, 2 - left, 3 - up
    # For rest, it doesn't mean anything
    door_location_mask = object_ids == _DOOR_OBJECT_ID
    door_states = torch.where(door_location_mask, states, 0)
    agent_location_mask = object_ids == _AGENT_OBJECT_ID
    agent_dir = torch.where(agent_location_mask, states, 0)

    object_dense_ids = _OBJECT_LOOKUP[object_ids]
    color_dense_ids = _COLOR_LOOKUP[color_ids]

    objects = F.one_hot(object_dense_ids, num_classes=6)
    colors = F.one_hot(color_dense_ids, num_classes=4)
    door_states = F.one_hot(door_states, num_classes=3)
    agent_dir = F.one_hot(agent_dir, num_classes=4)

    # Zero-out encoded states for all cells that dont correspond to either door or agent
    door_states[~door_location_mask] = 0
    agent_dir[~agent_location_mask] = 0

    encoded_obs = torch.cat((objects, colors, door_states, agent_dir), dim=2)
    return encoded_obs.flatten().float()


class DoorKeyDemoDataset(Dataset):
    def __init__(self, demo_data_dir: Path = Path("data", "demos"), seeds: list[int] | None = None):
        if seeds is not None:
            if not isinstance(seeds, list):
                raise TypeError("`seeds` has to be a list!")
            episode_files = [demo_data_dir / f"seed_{seed:04d}.jsonl" for seed in seeds]
        else:
            episode_files = sorted(demo_data_dir.glob("*.jsonl"))

        observations = []
        actions = []
        for episode_file in episode_files:
            episode_obs, episode_actions = self._process_episode_file(episode_file)
            observations.extend(episode_obs)
            actions.extend(episode_actions)

        self.observations: Tensor = torch.stack(observations)
        self.actions: Tensor = torch.stack(actions)

    def _process_episode_file(self, file: Path) -> tuple[list[Tensor], list[Tensor]]:
        observations = []
        actions = []
        with open(file) as f:
            for line in f:
                record = json.loads(line)
                if "obs" not in record or "action" not in record:
                    print(
                        f"Skipping record as its potentially malformed!! 'obs' and/or 'action' not "
                        f"found in keys - {record.keys()}"
                    )
                    continue

                observation = encode_obs(record["obs"])
                action = torch.tensor(record["action"], dtype=torch.long)
                observations.append(observation)
                actions.append(action)

        return observations, actions

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.observations[idx], self.actions[idx]


class DoorKeyPolicyModel(nn.Module):
    _NUM_ACTIONS = 7

    def __init__(self, input_dim: int = 6 * 6 * 17) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dim, 128)
        self.output = nn.Linear(128, self._NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.output(x)


def compute_accuracy(predictions: Tensor, actions: Tensor) -> float:
    """Fraction of examples where argmax(predictions) matches the expert action.

    A per-step imitation-accuracy proxy, not full-episode rollout success --
    see failure_spec.md's success definition (terminated=True, reward > 0) for
    the actual target metric, which this does not measure.
    """
    predicted_actions = torch.argmax(predictions, dim=1)
    correct = (predicted_actions == actions).sum().item()
    return correct / actions.size(0)


def _evaluate(model: nn.Module, dataloader: DataLoader) -> tuple[Tensor, Tensor]:
    """Runs `model` over `dataloader` in eval mode (no grad), returning the
    concatenated (predictions, actions) across all batches.
    """
    model.eval()
    all_predictions = []
    all_actions = []
    with torch.no_grad():
        for batch_observations, batch_actions in dataloader:
            all_predictions.append(model(batch_observations))
            all_actions.append(batch_actions)
    return torch.cat(all_predictions), torch.cat(all_actions)


def train_for_bc(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    num_epochs: int,
) -> None:
    for epoch in range(num_epochs):
        running_loss = 0.0

        model.train()
        for batch_observations, batch_actions in train_dataloader:
            optimizer.zero_grad()
            pred_batch_actions = model(batch_observations)
            loss = loss_fn(pred_batch_actions, batch_actions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_predictions, train_actions = _evaluate(model, train_dataloader)
        train_accuracy = compute_accuracy(train_predictions, train_actions)

        val_predictions, val_actions = _evaluate(model, val_dataloader)
        val_loss = loss_fn(val_predictions, val_actions).item()
        val_accuracy = compute_accuracy(val_predictions, val_actions)

        print(
            f"{epoch=} AvgTrainingLoss={running_loss / len(train_dataloader)} "
            f"TrainAccuracy={train_accuracy:.2%} "
            f"ValLoss={val_loss:.4f} ValAccuracy={val_accuracy:.2%}"
        )


if __name__ == "__main__":
    # Test
    raw_obs = [
        [[2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0]],
        [[2, 5, 0], [1, 0, 0], [1, 0, 0], [10, 0, 1], [1, 0, 0], [2, 5, 0]],
        [[2, 5, 0], [1, 0, 0], [1, 0, 0], [5, 4, 0], [1, 0, 0], [2, 5, 0]],
        [[2, 5, 0], [4, 4, 2], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0]],
        [[2, 5, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [8, 1, 0], [2, 5, 0]],
        [[2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0]],
    ]
    encoded_obs = encode_obs(raw_obs)
    print(f"Raw observation: {raw_obs}")
    print(f"Encoded observation: {encoded_obs}")

    test_seeds = [28, 4]
    data = DoorKeyDemoDataset(seeds=test_seeds)
    print(len(data.observations))
    print(data[:3])

    model = DoorKeyPolicyModel()
    out = model(data.observations[0])
    print(out.shape)
    print(out)
