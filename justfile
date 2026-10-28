# Project actions for pydantic-zarr. Run `just --list` to see all recipes.

# Default python / zarr versions for single-environment recipes.
# Override like `just py=3.13 zarr=3.0.10 test`.
py := "3.12"
zarr := "3.1.0"

# List available recipes
default:
    @just --list

# Install pre-commit hooks
setup:
    pre-commit install

# Run all linters and formatters via pre-commit
lint:
    pre-commit run --all-files --show-diff-on-failure

# Run pyrefly type checking
typecheck:
    hatch run types:check

# Run the test suite against one python/zarr combo, passing extra args to pytest
test *args:
    hatch run test.py{{ py }}-{{ zarr }}:test {{ args }}

# Run the test suite with coverage against one python/zarr combo
test-cov:
    hatch run test.py{{ py }}-{{ zarr }}:list-env
    hatch run test.py{{ py }}-{{ zarr }}:test-cov

# Run the test suite without zarr installed
test-base *args:
    hatch run test-base.py{{ py }}:test {{ args }}

# Run the test suite without zarr installed, with coverage
test-base-cov:
    hatch run test-base.py{{ py }}:list-env
    hatch run test-base.py{{ py }}:test-cov

# Run the test suite across the full python/zarr matrix
test-all:
    hatch run test:test

# Run the test suite without zarr installed across every python version
test-base-all:
    hatch run test-base:test

# Check that pydantic-zarr imports with only its required dependencies installed
test-min-reqs:
    uv run --isolated --no-project --python {{ py }} --with . --reinstall-package pydantic-zarr \
        python -c "import pydantic_zarr.v2; import pydantic_zarr.v3"

# Run the documentation tests (doctests)
test-docs:
    hatch run docs:test

# Build the documentation
docs-build:
    hatch run docs:build

# Serve the documentation locally, rebuilding on changes
docs-serve:
    hatch run docs:serve

# Check that changelog entries are named correctly
changelog-check:
    uv run --no-sync python ci/check_changelog_entries.py

# uv is pinned so a new uv release's lock-format or resolver changes cannot
# fail (or wrongly pass) unrelated pull requests.
# Check that uv.lock matches pyproject.toml
lock-check:
    uvx uv@0.13.5 lock --check

# Run every CI check locally (the test matrix without coverage)
ci: lint lock-check changelog-check typecheck test-docs test-min-reqs test-all test-base-all

# Build the sdist and wheel distributions
build:
    hatch build

# Remove build, test, and docs artifacts
clean:
    rm -rf dist site htmlcov .coverage*
