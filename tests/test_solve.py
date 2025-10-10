from pathlib import Path
import numpy as np
from grid_feedback_optimizer.models.loader import load_network
from grid_feedback_optimizer.engine.solve import solve
from power_grid_model import ComponentType



def test_solve_from_example():

    # Path to the example JSON in your project
    EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example.json"
    # Load the network
    network = load_network(EXAMPLE_JSON)
    output_data, gen_update, iterates = solve(network)

def test_solve_from_example_with_transformer():

    # Path to the example JSON in your project
    EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example_with_transformer.json"
    # Load the network
    network = load_network(EXAMPLE_JSON)
    output_data, gen_update, iterates = solve(network)
    
if __name__ == "__main__":
    test_solve_from_example()
    test_solve_from_example_with_transformer()
