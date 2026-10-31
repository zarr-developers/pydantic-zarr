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
| Preview the release notes for a version | `just release-notes 0.11.0 --draft` |
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
only `run:` steps in the CI workflows that are not `just` recipes install
hatch (the publish workflow, `cd.yml`, builds with hatch directly) or drive
the release (see [Releasing](#releasing)).

## Commit messages and pull request titles

Releases are cut by [release-please](https://github.com/googleapis/release-please),
which reads the commit messages on `main` to decide the next version, so they
must follow [Conventional Commits](https://www.conventionalcommits.org):
`<type>(<optional scope>): <description>`, with a `!` after the type (or
scope) for breaking changes. Pull requests are squash-merged with the pull
request title as the commit message and an empty commit body, so only the
title reaches `main`: it is the title that has to follow the convention, for
example `fix(v3): reject ...` or `feat(core)!: keep ...`. Footers in the
pull request's own commits (`BREAKING CHANGE:`, `Release-As:`) are dropped by
the squash, so mark a breaking change with `!` in the title.

While pydantic-zarr is below 1.0, the version is bumped like this:

| Commits since the last release include | Next version (from 0.10.0) |
| --- | --- |
| a breaking change (`feat!:`, `fix!:`, ...) | 0.11.0 |
| a `feat:`, `fix:` or `perf:` (and nothing breaking) | 0.10.1 |
| only other types (`docs:`, `ci:`, `chore:`, `test:`, ...) | no release |

To release a different version than the computed one, the maintainer
merging a pull request types a `Release-As: x.y.z` footer into the
description box of the squash-merge dialog (it starts empty). That puts the
footer in the commit body on `main`, where release-please reads it, and the
release pull request switches to that version.

## Changelog entries

Every user-facing change needs a [towncrier](https://towncrier.readthedocs.io)
fragment in `changes/`, named `<pull request number>.<type>.md` where `<type>`
is one of `feature`, `bugfix`, `doc`, `removal`, or `misc` (the authoritative list lives in `changes/README.md`). The fragment is one or more
complete sentences describing the change from a user's point of view; it
becomes part of the release notes.

The fragments, not the commit messages, are the release notes: release-please
only computes the version and opens the release pull request (it does not
write a changelog), and the release workflow folds the fragments into
`docs/release-notes.md` on that pull request. The commit messages decide
*whether* and *how far* the version moves; the fragments say *what* changed.

## Releasing

Releases are automated by `.github/workflows/release.yml`; nobody tags or
uploads by hand.

1. **The release pull request.** After every push to `main` that includes a
   releasable commit (see the table above), release-please opens or updates a
   pull request titled `chore(main): release <version>`. It bumps the version
   in `.release-please-manifest.json`, and the workflow then runs
   `just release-notes <version> --yes` on its branch, which moves every
   fragment in `changes/` into a new section of `docs/release-notes.md`. Each
   later push to `main` rebuilds the branch, so the pull request always
   describes everything merged so far.
2. **Review it.** Read the rendered release notes in the pull request's diff,
   and fix any fragment by editing it on `main` (in its own pull request), not
   on the release branch, which is regenerated. The release pull request is
   opened with the workflow's token, so CI does not run on it; it changes
   only the manifest, the release notes and the fragments, and the code it
   releases already passed CI on `main`.
3. **Merge it** (squash). release-please then tags the merge commit
   `v<version>` and creates the GitHub release.
4. **Publish.** The release workflow dispatches `cd.yml` on the new tag. It
   builds the sdist and wheel (hatch-vcs takes the version from the tag) and
   publishes them to PyPI through trusted publishing in the `release`
   environment, which waits for a maintainer to approve the deployment in
   the run's page on the Actions tab. If that run fails, rerun it from the
   Actions tab, or run `cd.yml` manually with the tag selected as the ref.

The version in `pyproject.toml` is never edited: it is derived from the git
tag at build time, and `.release-please-manifest.json` only records the last
released version for release-please.

## Dependencies and the lockfile

Library dependencies (and their floors) live in `pyproject.toml`. The
committed `uv.lock` pins the development and documentation environments; if
you change anything in `pyproject.toml`, run `uv lock` and commit the result —
CI checks that the lockfile is current. The test matrix intentionally does
not use the lockfile: it resolves each python/zarr cell fresh, so the library
is tested against what users would actually install.
