from pathlib import Path
import numpy as np
from grid_feedback_optimizer.io.loader import load_network
from grid_feedback_optimizer.engine.solve import solve

# Path to the example JSON in your project
EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example.json"

def test_solve_from_example():

    # Load the network
    network = load_network(EXAMPLE_JSON)
    output_data = solve(network)
    
if __name__ == "__main__":
    test_solve_from_example()
