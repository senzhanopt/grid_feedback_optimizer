from pathlib import Path
from grid_feedback_optimizer.io.loader import load_network
from grid_feedback_optimizer.engine.solve import solve
from grid_feedback_optimizer.utils.utils import print_component


def main(json_path: str, max_iter: int = 100, tol: float = 1e-4, print_iteration = False):
    """
    Run grid feedback optimizer from a JSON file path provided as string.
    """
    # Convert to Path
    json_path = Path(json_path)

    # Load network
    network = load_network(json_path)

    # Solve
    output_data, optimized_gen = solve(network, max_iter=max_iter, tol=tol, print_iteration = print_iteration)

    # Print final results
    print("==== Final Results ====")
    print_component(output_data, "node")
    print_component(output_data, "line")
    n_transformer = len(network.transformers)
    if n_transformer >= 1:
        print_component(output_data, "transformer")

    return output_data, optimized_gen


if __name__ == "__main__":
    # Example usage: user edits this line with their JSON path
    main("./examples/simple_example_with_transformer.json")

