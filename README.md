# Grid Feedback Optimizer

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**Grid Feedback Optimizer** is a Python package that uses **feedback control** to optimize generator and device setpoints in electrical distribution grids.
It reads a **JSON network description** and iteratively computes optimal setpoints.

This package is designed for experimenting with **voltage regulation** and **congestion management**, providing a flexible framework for feedback-based grid optimization.

## Features

* Load networks from JSON files.
* Iterative feedback-based optimization with configurable tolerance and maximum iterations.
* Print node and line results at the end of the run.
* Modular design (`io`, `engine`, `utils`) for extensions.

## Repository Structure

```
grid_feedback_optimizer/
src/
    main.py
    grid_feedback_optimizer/
        io/           # Loaders and I/O
        engine/       # Solver / optimization logic
        utils/        # Helper functions
examples/           # Example JSON network files
tests/              # Unit tests
requirements.txt    # Python dependencies
README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/senzhanopt/grid_feedback_optimizer.git
cd grid_feedback_optimizer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the optimizer with an example JSON network:

```bash
python src/main.py ./examples/simple_example.json
```

Optional parameters (via Python function call):

* `max_iter` (default = 100) — Maximum iterations
* `tol` (default = 1e-4) — Convergence tolerance
* `print_iteration` (default = False) — Print intermediate iteration results

**Python usage example:**

```python
from src.main import main

output_data, optimized_gen = main(
    json_path="./examples/simple_example.json",
    max_iter=200,
    tol=1e-5,
    print_iteration=True
)
```


## Testing

Run tests with:

```bash
pytest tests/
```

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Developed and maintained by **Sen Zhan**.
