from pathlib import Path
from grid_feedback_optimizer.io.loader import load_network
from grid_feedback_optimizer.io.writer import network_states_to_dict, setpoints_to_dict, save_results
from grid_feedback_optimizer.engine.powerflow import PowerFlowSolver
from power_grid_model import ComponentType
import numpy as np

def test_writer():

    # Path to the example JSON in your project
    EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example_with_transformer.json"

    # Load the network
    network = load_network(EXAMPLE_JSON)
    power_flow_solver = PowerFlowSolver(network)

    network_states_to_dict(power_flow_solver.base_output_data)
    gen_update = np.array([[1000.0, 500.0]])
    setpoints_to_dict(gen_update)
    save_results(power_flow_solver.base_output_data, gen_update)