from pathlib import Path
import numpy as np
from grid_feedback_optimizer.io.loader import load_network
from grid_feedback_optimizer.engine.powerflow import PowerFlowSolver

# Path to the example JSON in your project
EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example.json"

def test_powerflow_from_example():

    # Load the network
    network = load_network(EXAMPLE_JSON)

    power_flow_solver = PowerFlowSolver(network)

    power_flow_solver.run(gen_update =np.array([50000.0, 0.0]).reshape(1,2))