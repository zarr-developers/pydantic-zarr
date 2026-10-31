"""
Check that a pull request title follows Conventional Commits.

Pull requests are squash-merged with the title as the commit message, and
release-please derives the next version from those messages, so the title
decides whether (and how far) the version moves. The title is the first
argument, or the PR_TITLE environment variable when no argument is given.
"""

import os
import re
import sys

VALID_TYPES = [
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
]
# <type>(<optional scope>)<optional !>: <description>
TITLE_PATTERN = re.compile(r"(?P<type>[a-z]+)(\((?P<scope>[^()\s][^()]*)\))?!?: \S.*")


def problems(title: str) -> list[str]:
    match = TITLE_PATTERN.fullmatch(title)
    if match is None:
        return [
            "The title is not of the form `<type>(<optional scope>): <description>` "
            "(with a `!` before the colon for a breaking change)."
        ]
    if match["type"] not in VALID_TYPES:
        return [f"`{match['type']}` is not a valid type. Valid types are: {VALID_TYPES}"]
    return []


if __name__ == "__main__":
    title = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PR_TITLE", "")
    print(f"Checking pull request title: {title!r}")
    found = problems(title)
    if found:
        print("\n".join(found))
        print("See the 'Commit messages and pull request titles' section of docs/contributing.md.")
        sys.exit(1)

    sys.exit(0)
