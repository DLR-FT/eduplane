<!--
SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)

SPDX-License-Identifier: Apache-2.0
-->

# Extended Dubins Airplane (EDuPlane)

Welcome to **EDuPlane**, a Python package for *runtime-efficient trajectory planning for fixed-wing unmanned aircraft under consideration of flight performance constraints and uniform wind.*

This package contains reference implementations for multiple algorithms based upon and extending the [https://lavalle.pl/papers/ChiLav07b.pdf](Dubins airplane method).

The mathematical and algorithmic framework is presented in:

> S. Schopferer (2026) **Extensions to the Dubins Airplane Method - Flight Performance Constraints in Uniform Wind**  
> (accepted for publication in IEEE TAES, preprint available at [https://arxiv.org/abs/2611.08004])

If you use this software in an academic context, please cite this paper.

The notebook `notebooks/2026-TAES_ExtDubinsAirplane.ipynb` may serve as an entry point to reproduce the results presented in this paper and to explore the project.

## Installation

Install from source into a new virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

For development (tests, coverage, packaging, docs, notebooks), install additional dependencies:

```bash
pip install -e ".[dev]"
```

## Package structure

Top-level package structure:

- `model`: Kinematic and flight performance model implementation
- `traj`: Trajectory representations, both parametric and discrete
- `solver`: Planning and optimization algorithms for variations of the Dubins airplane problem
- `utils`: Various utility modules (plotting, logging, symbolic math functions and more)
- `docs`: API documentation (auto-generated)
- `notebooks`: Notebooks containing analysis of the algorithms, plots, and benchmarks

## Tests

Run the comprehensive test suite:
```bash
python3 -m unittest discover
# or
pytest tests/
```

The test suites for the solvers are quite comprehensive and may take a while to complete. Set the `NUM_TEST_CASES` environment variable to limit or increase the number of randomized test cases used to test the solvers and variants thereof (default is 100).

To produce a **test coverage** report:

```bash
pytest --cov=. tests/ --cov-report xml:cov.xml
```

To **profile** a specific test suite:

```bash
python3 -m cProfile -o profile.pstats -m unittest tests.solvers.test_dubins_path
snakeviz profile.pstats
```

For running **benchmarks**, force CPU performance mode:

```bash
# Install cpupower tool and show current config
sudo apt install linux-tools-common linux-tools-$(uname -r)
cpupower frequency-info

# Set performance mode
sudo cpupower frequency-set -g performance

# ... run benchmarks

# Set powersave mode
sudo cpupower frequency-set -g powersave
```

To run various linters, checks, unit tests and build the docs with a single command, run the `scripts/ci_tests.sh` script.

## Documentation

The API documentation is built with [Sphinx](https://www.sphinx-doc.org)
(MyST-Parser for Markdown, Furo theme).

Install the docs dependencies:

```bash
pip install -e ".[docs]"
```

Build the HTML docs:

```bash
# From the repository root
sphinx-build -b html docs/source docs/build/html

# ...or via the Makefile (also regenerates the API stubs)
cd docs && make html
```

Open `docs/build/html/index.html` in a browser afterwards. To treat warnings as
errors (recommended before publishing):

```bash
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

The `docs/Makefile` regenerates the API reference stubs with `sphinx-apidoc`
before every build (`make apidoc` to run it on its own), so the stubs cannot
drift out of sync with the package.

## Dependencies

Dependencies are declared in `pyproject.toml`.
License information for all runtime dependencies can be found in `THIRD-PARTY-NOTICES.md`.

## License

Licensed under the Apache License, Version 2.0. 
See the `LICENSES/Apache-2.0.txt` file for the full license text.
