# Grid Feedback Optimizer

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**Grid Feedback Optimizer** is a Python package that uses **feedback optimization** to optimize generator and device setpoints in electrical distribution grids.
It reads a **JSON network description** and iteratively computes optimal setpoints.

This package is designed for experimenting with **voltage regulation** and **congestion management**, providing a flexible framework for feedback-based grid optimization.

## Features

* Load and simulate networks from JSON files.
* Iterative feedback optimization using:
    - **gradient projection (GP)** algorithm,
    - **primal-dual (PD)** algorithm.
* Structured input and output data.
* Modular design (`models`, `engine`, `utils`) for extensions.

## Repository Structure

```
grid_feedback_optimizer/
src/
    grid_feedback_optimizer/
        model/        # Loaders and I/O
        engine/       # Power flow / optimization logic
        utils/        # Helper functions
        main.py
examples/           # Example JSON network files
tests/              # Tests
requirements.txt    # Python dependencies
README.md
```

## ⚙️ Installation

**Clone the repository:**

```bash
git clone https://github.com/senzhanopt/grid_feedback_optimizer.git
cd grid_feedback_optimizer
```

**Install dependencies for all parts:**
```bash
pip install -r requirements.txt
```

**Install the project in editable mode**
```bash
pip install -e .
```

## Usage


**Python usage example:**

```python
from grid_feedback_optimizer.models.loader import load_network
from grid_feedback_optimizer.engine.solve import solve
from grid_feedback_optimizer.engine.powerflow import PowerFlowSolver

# Load network from example JSON
network = load_network("../examples/simple_example_with_transformer.json")

# Initialize and check power flow
power_flow_solver = PowerFlowSolver(network)

# Run optimization using the Gradient Projection (GP) algorithm
res_gp = solve(network, algorithm="gp")

# Display and store results
res_gp.print_summary()
res_gp.plot_iterations()
res_gp.save("gp_result.json")

```


## License

This project is licensed under the [MIT License](LICENSE).

## Author

Developed and maintained by **Sen Zhan**  
📧 Email: [sen.zhan@outlook.com](mailto:sen.zhan@outlook.com)