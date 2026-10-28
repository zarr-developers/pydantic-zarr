# Contributing

Development uses [hatch](https://hatch.pypa.io) for environments,
[just](https://just.systems) as the task runner, [uv](https://docs.astral.sh/uv/)
for dependency resolution, and [prek](https://prek.j178.dev) for
linting. Install those four tools, then:

```sh
just setup      # install the git hooks
just test       # run the test suite on the default python/zarr combination
just --list     # see every available task
```

## Common tasks

| Task | Command |
| --- | --- |
| Run the tests on one python/zarr combination | `just test` (override with `just py=3.13 zarr=3.0.10 test`) |
| Run the tests without zarr installed | `just test-base` |
| Run the whole python/zarr matrix | `just test-all` |
| Type checking (pyrefly) | `just typecheck` |
| Lint and format everything | `just lint` |
| Run the tests with coverage | `just test-cov` (without zarr: `just test-base-cov`) |
| Validate changelog fragment names | `just changelog-check` |
| Check that `uv.lock` is current | `just lock-check` |
| Check the import with only required dependencies | `just test-min-reqs` |
| Run every CI check locally | `just ci` |
| Documentation tests | `just test-docs` |
| Build or serve the documentation | `just docs-build` / `just docs-serve` |

## Continuous integration

Every CI job runs one of the recipes above, so a CI failure can be
reproduced by running the same `just` command locally. The test matrix job
runs `just py=<python> zarr=<zarr> test-cov` for each cell (and
`just py=<python> test-base-cov` for the cell without zarr). `just ci` runs
all of the checks in one go, using the test matrix without coverage. The
only `run:` steps in the workflows that are not `just` recipes install
hatch.

## Changelog entries

Every user-facing change needs a [towncrier](https://towncrier.readthedocs.io)
fragment in `changes/`, named `<pull request number>.<type>.md` where `<type>`
is one of `feature`, `bugfix`, `doc`, `removal`, or `misc` (the authoritative list lives in `changes/README.md`). The fragment is one or more
complete sentences describing the change from a user's point of view; it
becomes part of the release notes.

## Dependencies and the lockfile

Library dependencies (and their floors) live in `pyproject.toml`. The
committed `uv.lock` pins the development and documentation environments; if
you change anything in `pyproject.toml`, run `uv lock` and commit the result —
CI checks that the lockfile is current. The test matrix intentionally does
not use the lockfile: it resolves each python/zarr cell fresh, so the library
is tested against what users would actually install.
