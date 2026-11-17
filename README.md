# Your LLM's p90 might not exist

Companion code for measuring three outcomes together:

- task resolution within two attempts
- p90 duration among resolved tasks
- resolution within a 30-second budget

A latency percentile excludes tasks without a duration event. Read resolution first, then latency among successes.

## Setup

```bash
uv sync --frozen
uv run pytest
uv run ruff check .
cp .env.example .env
```

Add the credentials listed in `.env.example`. Follow the accompanying tutorial to create the Agent Management config, metrics, and draft experiment.

## Verify the workflow

Run one row before the full paired pilot:

```bash
uv run python run_evaluation.py --tasks 1
uv run python run_evaluation.py
```

After checking that events reach the intended environment and starting the experiment:

```bash
uv run python run_experiment.py --tasks 170 --confirm-paid-run
```

The paid runner requires explicit confirmation and refuses to overwrite an existing results log.

## Reproducibility

`data/provenance.json` pins the BFCL source and task split. `protocol.json` records the design. The checked-in confirmatory log, measured aggregates, and LaunchDarkly resource snapshot make the published result auditable from a fresh clone.
