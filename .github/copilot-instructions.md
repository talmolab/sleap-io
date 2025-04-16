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