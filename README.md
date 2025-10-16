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
### Grid components

Follow [power-grid-model](https://power-grid-model.readthedocs.io/en/stable/user_manual/components.html) for definition of buses (nodes), lines, transformers, and sources. 

### RenewGen

`RenewGen` models **controllable generators and power-consuming devices**.

- **Generator:** `p_max > 0` and `p_min >= 0`  
- **Load:** `p_max < 0` and `p_min <= 0`  
- **Flexible device:** `p_min < 0 < p_max` (can generate or consume)  

**Key attributes:**

- `index`, `bus`: identifiers  
- `p_max`, `p_min`: active power limits  
- `s_inv`: apparent power rating  
- `p_norm`: normal active power (auto-computed if not set)  
- `q_norm`: normal reactive power (0.0 if not set)
- `c1_p`: linear active power cost coefficients  
- `c2_p`: quadrtic active power cost coefficients for deviation from `p_norm`
- `c1_q`: linear reactive power cost coefficients  
- `c2_q`: quadrtic reactive power cost coefficients for deviation from `q_norm`

**Minimization cost function:**  

Cost = c1_p × p + c2_p × (p - p_norm)² + c1_q × q + c2_q × (q - q_norm)²  

where `p` and `q` are the actual active and reactive power outputs.

`p_norm` is computed automatically:  
- Generator → `p_norm = p_max`  
- Load → `p_norm = p_min`  
- Flexible → `p_norm = 0`

### Load

`Load` models **non-controllable units**, either a generator or a load.

- **Load:** `p_norm >= 0` 
- **Generator:** `p_norm < 0`

**Key attributes:**

- `index`, `bus`: identifiers  
- `p_norm`, `q_norm`: active and reactive power

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Developed and maintained by **Sen Zhan**  
📧 Email: [sen.zhan@outlook.com](mailto:sen.zhan@outlook.com)