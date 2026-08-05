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
    pre-commit run --all-files

# Run mypy type checking
typecheck:
    hatch run types:check

# Run the test suite against one python/zarr combo, passing extra args to pytest
test *args:
    hatch run test.py{{ py }}-{{ zarr }}:test {{ args }}

# Run the test suite with coverage against one python/zarr combo
test-cov:
    hatch run test.py{{ py }}-{{ zarr }}:test-cov

# Run the test suite without zarr installed
test-base *args:
    hatch run test-base.py{{ py }}:test {{ args }}

# Run the test suite across the full python/zarr matrix
test-all:
    hatch run test:test

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

# Build the sdist and wheel distributions
build:
    hatch build

# Remove build, test, and docs artifacts
clean:
    rm -rf dist site htmlcov .coverage*
