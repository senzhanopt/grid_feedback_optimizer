from pathlib import Path
import numpy as np
from grid_feedback_optimizer.models.loader import load_network
from grid_feedback_optimizer.engine.powerflow import PowerFlowSolver



def test_powerflow_from_example():

    # Path to the example JSON in your project
    EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example.json"

    # Load the network
    network = load_network(EXAMPLE_JSON)
    power_flow_solver = PowerFlowSolver(network)
    power_flow_solver.run(gen_update =np.array([50000.0, 0.0]).reshape(1,2))
    power_flow_solver.obtain_sensitivity()

    assert power_flow_solver.is_congested == True



def test_powerflow_from_example_with_transformer():

    # Path to the example JSON in your project
    EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example_with_transformer.json"

    # Load the network
    network = load_network(EXAMPLE_JSON)
    power_flow_solver = PowerFlowSolver(network)
    power_flow_solver.run(gen_update =np.array([50000.0, 0.0]).reshape(1,2))
    power_flow_solver.obtain_sensitivity()

    assert power_flow_solver.is_congested == True