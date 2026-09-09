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

## Testing

```bash
uv run pytest
```
