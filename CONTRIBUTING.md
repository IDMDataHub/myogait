# Contributing to myogait

Thanks for your interest in myogait. Bug reports, questions, documentation
fixes and code contributions are all welcome.

## Reporting a problem or asking a question

Open an [issue](https://github.com/IDMDataHub/myogait/issues) using the bug
report, feature request or question template. For a bug, please include the
myogait version (`python -c "import myogait; print(myogait.__version__)"`),
your operating system and Python version, the pose backend, and a minimal way
to reproduce it (a short clip or a `.myogait.json` file if you can share one).
Do not upload identifiable patient videos.

## Contributing code

1. Fork the repository and create a branch from `master`.
2. Install in development mode:

   ```bash
   pip install -e ".[dev]"
   ```

3. Make your change, with tests in `tests/` for any new behaviour or fix.
4. Check that the test suite and the linter pass, as in continuous integration:

   ```bash
   pytest
   ruff check myogait/ tests/ --select E,W,F --ignore E501
   ```

5. Add a line to `CHANGELOG.md` under an "Unreleased" heading and open a pull
   request describing what changes and why.

Keep the public API backward compatible where possible; the project follows
semantic versioning. Changes that alter validated outputs (angles, events,
spatio-temporal parameters) should say how they were checked against a
reference (see [`validation/`](validation/README.md)).

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you agree to uphold it.

## Contact

Frédéric Fer, Institut de Myologie — f.fer@institut-myologie.org
