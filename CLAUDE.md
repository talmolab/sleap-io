# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

sleap-io is a standalone utility library for working with animal pose tracking data. It provides:
- Reading/writing pose tracking data in various formats (SLEAP, NWB, Label Studio, JABS)
- Data structure manipulation and conversion
- Video I/O operations
- Minimal dependencies (no labeling, training, or inference functionality)

## Code Style Requirements

1. **Formatting**: Use `ruff format` with max line length of 88
2. **Docstrings**: Google style, document "Attributes" section in class-level docstring
3. **Type hints**: Always include for function arguments and return types
4. **Import order**: Standard library, third-party, local (enforced by ruff)

## Development workflow

1. Never work in `main`, always create a scoped feature branch (e.g., `feature/x` or `fix/y` if not specified explicitly).
2. After creating the branch, make sure to plan for the tasks you will be doing.
3. When experimenting and introspecting, feel free to create a directory in `tmp/{BRANCH_NAME}` for prototyping, exploration, and note-taking. This will not be checked into the git history.
4. When committing, never commit all changes. Always carefully analyze the files and hunks that changed and create specific and multiple PRs to avoid committing unintended work, or bundling changes together that would make it hard to revert them.
5. After making your changes, be sure to lint and maximize coverage.
6. If there are enhancements or API changes, be sure to update the `docs/` appropriately.
7. Create a PR using the `gh` CLI when the changes are close to ready.
8. When finishing up, use `gh` to check the CI every 30 seconds until it finishes running and check the logs if CI failed.
9. When the user says to go ahead and merge, then squash merge with `gh pr merge --delete-branch`.

### PR Descriptions

When updating PR descriptions (from .claude/commands/pr-description.md):
1. Fetch current PR metadata and linked issues using `gh` CLI
2. Include: Summary, Key Changes, Example Usage, API Changes, Testing, and Design Decisions
3. Document reasoning behind implementation choices for future reference

## Testing Rules

1. Use existing fixtures from `tests/fixtures/` when possible. Read the modules there to learn about relevant fixtures.
2. Create minimal synthetic data for new tests rather than files when there is no appropriate existing fixture.
3. Use `tmp_path` for any I/O operations in tests.
4. Write multiple focused tests rather than one complex test.
5. Place tests in corresponding module under `tests/` (e.g., `sleap_io/io/slp.py` → `tests/io/test_slp.py`)
6. Never create new test modules unless a new package module was created.
7. When adding tests, use global imports at the module-level rather than importing locally within a test function unless strictly needed (e.g., for import checking). Analyze current imports to find the best place to add the import statement and do not duplicate existing imports.
8. Use `pytest` function-style testing rather than `unittest`-style tests.
9. **NEVER** use mocking or monkey patching as a quick fix for increasing coverage. If the condition or execution branch can't be reproduced (e.g., defensive coding, rare edge cases), re-evaluate whether that condition should exist. Code branches that are difficult to test are an anti-pattern.

## Common Development Tasks

### Adding a New I/O Format
1. Create module in `sleap_io/io/` with reader/writer functions
2. Add format detection to `sleap_io/io/main.py`
3. Create comprehensive tests in `tests/io/`
4. Update documentation with format specifications