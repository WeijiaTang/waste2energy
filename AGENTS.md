# Repository Guidelines

## Project Structure & Module Organization
Core package code lives in `src/waste2energy/`. Main domains are `planning/`, `scenarios/`, `operation/`, `surrogates/`, `models/`, and shared helpers in `common/` and `data/`. Tests live in `tests/` and follow the package layout with smoke and unit coverage. Research scripts are under `scripts/` (`data-process/`, `data-crawl/`, `plot/`). Input datasets live in `data/raw/`; reproducible derived tables belong in `data/processed/`. Generated run artifacts go to `outputs/` and manuscript material lives in `docs/` plus `waste2energy-paper/`.

## Build, Test, and Development Commands
- `uv sync --dev` — create/update the preferred development environment.
- `python -m pip install -e .` — editable install fallback if `uv` is unavailable.
- `pytest` — run the full test suite defined in `pyproject.toml`.
- `pytest tests/test_planning_smoke.py -q` — quick regression check for planning changes.
- `waste2energy-plan` — run the baseline planning workflow.
- `waste2energy-benchmark` / `waste2energy-scenario` / `waste2energy-audit` — run benchmark, robustness, and audit pipelines.
- `python scripts/data-process/11_build_planning_mult_pathway_dataset.py` — refresh the optimization-ready planning dataset.

## Coding Style & Naming Conventions
Use Python 3.11+ with 4-space indentation, UTF-8, and type hints for new public functions. Follow the existing import grouping and concise, Black-friendly line wrapping even though no formatter is pinned in `pyproject.toml`. Use `snake_case` for modules, functions, variables, CLI flags, and test files; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants. Keep generated filenames descriptive, e.g. `scenario_summary.csv` or `run_config.json`.

## Testing Guidelines
Use `pytest` and place tests in `tests/test_*.py`. Prefer small unit tests for scoring, constraints, and data transforms, plus smoke tests when changing CLI workflows or artifact generation. Every bug fix should include a regression test covering the affected module or command path.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `tests`, `add results and plot`, or focused data updates. Prefer one topic per commit and use an optional scope when helpful, e.g. `planning: tighten solver fallback`. PRs should include: purpose, key files changed, commands run (`pytest`, relevant CLI), and any affected artifact paths, tables, or figures. Link the relevant issue, task, or spec file when available.

## Data & Artifact Hygiene
Do not hand-edit generated files in `outputs/` unless the task is explicitly about fixtures or golden outputs. When changing pipeline logic, regenerate artifacts from code and record the script or CLI command used.
