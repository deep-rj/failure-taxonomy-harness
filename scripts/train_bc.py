"""Train the BC policy MLP on the demo seed data.

Splits the demo seeds (failure_taxonomy_harness.seeds.DEMO_SEEDS, per
failure_spec.md §4) into a train/val set by seed (not by individual step) so
that no episode's steps leak across the
split -- see failure_spec.md §4's own demo/eval seed-collision discussion for
why splitting at the step level would be a leakage bug.

The policy is a deliberately weakened fixture (failure_spec.md §3, 40-60%
success band), tuned primarily via `--num-epochs`.

Usage:
    uv run python scripts/train_bc.py
    uv run python scripts/train_bc.py --num-epochs 3 --batch-size 16
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from failure_taxonomy_harness.bc import DoorKeyDemoDataset, DoorKeyPolicyModel, train_for_bc
from failure_taxonomy_harness.seeds import DEMO_SEEDS

DEFAULT_DEMO_DIR = Path("data/demos")
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_SPLIT_SEED = 0
DEFAULT_NUM_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_MODEL_OUT = Path("models/bc_policy.pt")


def split_seeds(seeds, val_split: float, split_seed: int) -> tuple[list[int], list[int]]:
    """Shuffles `seeds` deterministically (via `split_seed`) and splits off
    `val_split` fraction into a val set, returning (train_seeds, val_seeds).
    """
    shuffled = list(seeds)
    random.Random(split_seed).shuffle(shuffled)  # noqa: S311
    num_val = round(len(shuffled) * val_split)
    val_seeds = shuffled[:num_val]
    train_seeds = shuffled[num_val:]
    return train_seeds, val_seeds


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    args = parser.parse_args()

    train_seeds, val_seeds = split_seeds(DEMO_SEEDS, args.val_split, args.split_seed)
    print(f"train_seeds ({len(train_seeds)}) = {sorted(train_seeds)}")
    print(f"val_seeds   ({len(val_seeds)}) = {sorted(val_seeds)}")

    train_dataset = DoorKeyDemoDataset(args.demo_dir, seeds=train_seeds)
    val_dataset = DoorKeyDemoDataset(args.demo_dir, seeds=val_seeds)
    print(f"train samples={len(train_dataset)}  val samples={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    model = DoorKeyPolicyModel(input_dim=train_dataset.observations.shape[1])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_for_bc(model, train_loader, val_loader, loss_fn, optimizer, args.num_epochs)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_out)


if __name__ == "__main__":
    main()
