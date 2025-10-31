# Contributing to OKAPI

OKAPI is an open-source project, and contributions are welcome. This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/damiankucharski/OKAPI.git
   cd OKAPI
   ```
1.1 Or fork it to be able to send pull requests

2. Install uv if you don't have it already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Sync the environment with all dependencies:
   ```bash
   uv sync --all-groups --all-extras
   ```

   This will automatically:
   - Create a virtual environment if needed
   - Update the lock file if necessary
   - Install the project in editable mode
   - Install all dependencies including development dependencies and optional extras

4. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Code Style

OKAPI uses:

- [Ruff](https://github.com/charliermarsh/ruff) for code linting and formatting
- [mypy](https://mypy.readthedocs.io/) for type checking

Docstrings should follow the Google-style format, which is used throughout the codebase.

## Testing

Run the tests pytest:

```bash
make test
```

Check typing with mypy:

```bash
make mypy
```

Both of these can be run with the following make command:

```bash
make test_all
```

When adding new features, please include tests. If you do not have GNU Make installed, you can just run the commands from the Makefile yourself.

## Documentation

Documentation is written in Markdown and built using MkDocs with the Material theme. API documentation is automatically generated from docstrings using mkdocstrings.

To preview the documentation locally:

```bash
make serve_docs
```

Then visit `http://127.0.0.1:8000` in your browser.

## Pull Request Process

1. Fork the repository
2. Create a feature branch for your changes
3. Make your changes
4. Run the tests and make sure they pass
5. Update documentation as needed
6. Submit a pull request

## Reporting Issues

If you find a bug or have a feature request, please create an issue in the GitHub repository. Please include:

- A clear and descriptive title
- A description of the issue or feature request
- Steps to reproduce the issue (for bugs)
- Any relevant code samples, error messages, or screenshots
