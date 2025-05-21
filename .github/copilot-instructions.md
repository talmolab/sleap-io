Use Google style docstrings and `black` formatting with max line length of 88.

Type hint arguments and return types in functions and methods.

Prefer documenting an "Attributes" section in the class-level docstring over documenting the constructor.

When running tests, don't forget to activate the conda environment first (check `environment.yml` or infer the name from context).

When creating tests, either use existing fixtures or define minimal testing data within the test itself.

Opt for creating more tests for different cases over baking in many conditions into the same test.

When maximizing test coverage, use the following command to check for line-by-line coverage:

```
conda activate {ENV_NAME} && pytest {TEST_MODULE} -v --cov={PACKAGE_NAME} --cov-report=json && coverage annotate --include="*/{MODULE_NAME}"
```

*Example:* `conda activate sleap-io && pytest tests/model/test_labels.py -v --cov=sleap_io --cov-report=json && coverage annotate --include="*/sleap_io/model/labels.py"`

This will produce a file called `{MODULE_NAME},cover`. Parse it to check which lines are being covered. Confirm the coverage by reporting the number of lines covered and not covered in each function you are working on.

When writing tests, be sure to use `pytest` best practices, including doing I/O in `tmp_path` directories.

Do not create new test modules unless a new package module was created. Place relevant tests in the corresponding test module in the `tests/` subfolder.

Prefer to create minimal synthetic data for tests, but if needed use real data from **existing** fixtures. Look at other tests, especially in the module you're writing tests in, for examples of how to use existing fixtures or how to create dummy data structures.

At the beginning of a chat session, acknowledge that you have read these instructions by saying: "Using instructions from `.github/copilot-instructions.md`."

At the end of a session when the request appears complete (for example, in Agent mode), confirm that you have tested for coverage and report the number of lines covered in the modules you worked on.