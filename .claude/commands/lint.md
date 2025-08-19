Run linting with `ruff`.

Command:

```
uv run ruff format sleap_io tests && uv run ruff check --fix sleap_io tests
```

Then manually fix any remaining errors which cannot be automatically fixed by ruff.