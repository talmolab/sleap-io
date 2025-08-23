Run tests with coverage.

Command to run:
```
uv run pytest -q --maxfail=1 --cov --cov-branch && rm .coverage.* && uv run coverage annotate
```

This generates a coverage annotation file next to each module with the name `{module_name.py},cover`, as well as a simple summary.

To get the final actionable summary, run this script:

```
uv run python scripts/cov_summary.py --only-pr-diff-lines
```

This will output one module per line with line number ranges for missing coverage. Importantly, it will filter it by diffs in the PR.

Use this summary together with the corresponding `,cover` file to describe each miss to inform subsequent test development.