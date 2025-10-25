# Grid Feedback Optimizer

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**Grid Feedback Optimizer** is a Python package that uses **feedback optimization** to optimize generator and device setpoints in electrical distribution grids.
It reads a **JSON/EXCEL network description** and iteratively computes optimal setpoints.

This package is designed for experimenting with **voltage regulation** and **congestion management**, providing a flexible framework for feedback-based grid optimization.

## Features

* Load and simulate networks from JSON/EXCEL files.
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
    examples/           # Example JSON/EXCEL network files
    tests/              # Tests
    requirements.txt    # Python dependencies
    README.md
```

## ⚙️ Installation

**Install from PyPI**

```bash
pip install grid-feedback-optimizer
```

## 🐳 Run with Docker

You can run this project using the pre-built Docker image:

```bash
docker pull senzhan2025/grid-feedback-optimizer:latest
docker run --rm -v $(pwd)/output:/app/output senzhan2025/grid-feedback-optimizer:latest examples/simple_example.json --save_path output/result.json --verbose
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


### 🧩 `solve()` Function Parameters

The `solve()` function performs the iterative feedback optimization between power flow calculation and control updates.  
It supports both **Gradient Projection (GP)** and **Primal-Dual (PD)** algorithms.

```python
solve(
    network: Network,
    max_iter: int = 1000,
    tol: float = 1e-3,
    delta_p: float = 1.0,
    delta_q: float = 1.0,
    algorithm: str = "gp",
    alpha: float = 0.5,
    alpha_v: float = 10.0,
    alpha_l: float = 10.0,
    alpha_t: float = 10.0,
    record_iterates: bool = True,
    solver: str = "CLARABEL",
    loading_meas_side: str = "from",
    rel_tol: float = 1E-4,
    rel_tol_line: float = 1E-2,
    **solver_kwargs
)
```

#### **Parameters**

| Parameter | Type | Default | Description |
|------------|------|----------|--------------|
| **`network`** | `Network` | — | Grid model object containing nodes, lines, transformers, loads, and generators (loaded via `load_network` or `load_network_from_excel`). |
| **`max_iter`** | `int` | `1000` | Maximum number of optimization–power flow iterations. |
| **`tol`** | `float` | `1e-3` | Convergence tolerance for generator setpoint changes between iterations. |
| **`delta_p`** | `float` | `1.0` | Small perturbation (in W) used for computing **active power sensitivities**. |
| **`delta_q`** | `float` | `1.0` | Small perturbation (in VAR) used for computing **reactive power sensitivities**. |
| **`algorithm`** | `str` | `"gp"` | Optimization algorithm to use: <br>• `"gp"` → Gradient Projection <br>• `"pd"` → Primal-Dual |
| **`alpha`** | `float` | `0.5` | Step size (learning rate) for generator setpoint updates (used in both GP and PD). |
| **`alpha_v`** | `float` | `10.0` | Voltage-related dual variable step size (only used in PD algorithm). |
| **`alpha_l`** | `float` | `10.0` | Line-loading-related dual variable step size (only used in PD algorithm). |
| **`alpha_t`** | `float` | `10.0` | Transformer-loading-related dual variable step size (only used in PD algorithm). |
| **`record_iterates`** | `bool` | `True` | If `True`, stores all intermediate iteration data (useful for analysis and plotting). |
| **`solver`** | `str` | `"CLARABEL"` | Convex optimization solver backend for subproblems (e.g., `"CLARABEL"`, `"OSQP"`, `"SCS"`). |
| **`loading_meas_side`** | `str` | `"from"` | Defines which end of the line or transformer is used for measuring loading: `"from"` or `"to"`. |
| **`rel_tol`** | `float` | `"1E-4"` | Relative tolerance for sensitivity matrices other than `dP_line_dq` or `dQ_line_dp`. |
| **`rel_tol_line`** | `float` | `"1E-2"` | Relative tolerance for sensitivity matrices `dP_line_dq` and `dQ_line_dp`. |
| **`**kwargs`** | - | - | Optional solver parameters (e.g., `"verbose"`, `"BarHomogeneous"`). |

#### **Returns**

| Output | Type | Description |
|---------|------|-------------|
| **`SolveResults`** | `dataclass` | Object containing: <br>• `final_output` — final power flow results <br>• `final_gen_update` — optimized generator setpoints <br>• `iterations` — list of all recorded iteration states (if `record_iterates=True`) |

---

🌀 **Conceptual workflow:**

1. **Power flow calculation** → compute voltages, line, and transformer loadings.  
2. **Optimization step** → update generator active/reactive power setpoints using feedback and sensitivities.  
3. **Iterate** until generator updates converge within `tol`.

The process continues until steady-state optimal operation is achieved under the given constraints.

# References

- **Gradient Projection** – V. Haberle, A. Hauswirth, L. Ortmann, S. Bolognani, and F. Dorfler, “Non-Convex Feedback Optimization with Input and Output Constraints,” *IEEE Control Systems Letters*, vol. 5, no. 1, pp. 343–348, 2021. [DOI: 10.1109/LCSYS.2020.3002152](https://doi.org/10.1109/LCSYS.2020.3002152)  
- **Primal-Dual** – E. Dall’Anese and A. Simonetto, “Optimal Power Flow Pursuit,” *IEEE Transactions on Smart Grid*, vol. 9, no. 2, 2018. [DOI: 10.1109/TSG.2016.2571982](https://doi.org/10.1109/TSG.2016.2571982)



## License

This project is licensed under the [MIT License](LICENSE).

## Author

Developed and maintained by **Sen Zhan**  
📧 Email: [sen.zhan@outlook.com](mailto:sen.zhan@outlook.com)