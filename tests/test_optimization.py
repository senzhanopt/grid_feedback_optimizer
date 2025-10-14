from pathlib import Path
import numpy as np
from grid_feedback_optimizer.models.loader import load_network
from grid_feedback_optimizer.engine.powerflow import PowerFlowSolver
from grid_feedback_optimizer.engine.grad_proj_optimizer import GradientProjectionOptimizer

def test_optimization_from_example():

    # Path to the example JSON in your project
    EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example.json"
    
    # Load the network
    network = load_network(EXAMPLE_JSON)
    power_flow_solver = PowerFlowSolver(network)
    sensitivities = power_flow_solver.obtain_sensitivity()
    optimizer = GradientProjectionOptimizer(network, sensitivities)

    param_dict = {
        "u_pu_meas": np.array([1.0,1.08]),
        "P_line_meas": np.array([-30000.0]),
        "Q_line_meas": np.array([30000.0]),
        "p_gen_last": np.array([60000.0]),
        "q_gen_last": np.array([0.0]),
    }

    optimizer.solve_problem(param_dict)

def test_optimization_from_example_with_transformer():

    # Path to the example JSON in your project
    EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example_with_transformer.json"
    
    # Load the network
    network = load_network(EXAMPLE_JSON)
    power_flow_solver = PowerFlowSolver(network)
    sensitivities = power_flow_solver.obtain_sensitivity()
    optimizer = GradientProjectionOptimizer(network, sensitivities)

    param_dict = {
        "u_pu_meas": np.array([1.0,1.0,1.08]),
        "P_line_meas": np.array([-30000.0]),
        "Q_line_meas": np.array([30000.0]),
        "P_transformer_meas": np.array([-30000.0]),
        "Q_transformer_meas": np.array([30000.0]),
        "p_gen_last": np.array([60000.0]),
        "q_gen_last": np.array([0.0]),
    }

    optimizer.solve_problem(param_dict)