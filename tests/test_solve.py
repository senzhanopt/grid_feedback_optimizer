from pathlib import Path
import numpy as np
from grid_feedback_optimizer.models.loader import load_network, load_network_from_excel
from grid_feedback_optimizer.engine.solve import solve



def test_solve_from_example():

    # Path to the example JSON in your project
    EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example.json"
    # Load the network
    network = load_network(EXAMPLE_JSON)
    res = solve(network)
    res.print_summary()

def test_solve_from_example_with_transformer():

    # Path to the example JSON in your project
    EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example_with_transformer.json"
    # Load the network
    network = load_network(EXAMPLE_JSON)
    res = solve(network)
    res.print_summary()
    #res.plot_iterations()

def test_solve_from_example_with_transformer_from_excel():

    # Path to the example JSON in your project
    EXAMPLE_EXCEL = Path(__file__).parent.parent / "examples" / "simple_example_with_transformer.xlsx"
    
    # Load the network
    network = load_network_from_excel(EXAMPLE_EXCEL)
    res = solve(network)
    res.print_summary()
    #res.plot_iterations()
    
if __name__ == "__main__":
    test_solve_from_example()
    test_solve_from_example_with_transformer()
    test_solve_from_example_with_transformer_from_excel()
