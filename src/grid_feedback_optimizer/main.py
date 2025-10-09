from pathlib import Path
from grid_feedback_optimizer.io.loader import load_network
from grid_feedback_optimizer.engine.solve import solve
from grid_feedback_optimizer.utils.utils import print_component


def main(json_path: str, max_iter: int = 100, tol: float = 1e-4, 
         delta_p: float = 1.0, delta_q: float = 1.0, alpha: float = 0.5, print_iteration = False):
    """
    Run grid feedback optimizer from a JSON file path provided as string.
    """
    # Convert to Path
    json_path = Path(json_path)

    # Load network
    network = load_network(json_path)

    # Solve
    output_data, optimized_gen = solve(network, max_iter = max_iter, tol = tol, 
                                       delta_p = delta_p, delta_q = delta_q,
                                       alpha = alpha, print_iteration = print_iteration)

    # Print final results
    print("==== Final Results ====")
    print_component(output_data, "node")
    print_component(output_data, "line")
    n_transformer = len(network.transformers)
    if n_transformer >= 1:
        print_component(output_data, "transformer")

    return output_data, optimized_gen



