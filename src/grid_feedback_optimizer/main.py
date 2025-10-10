from pathlib import Path
from grid_feedback_optimizer.models.loader import load_network
from grid_feedback_optimizer.engine.solve import solve


def main(json_path: str, max_iter: int = 100, tol: float = 1e-4, 
         delta_p: float = 1.0, delta_q: float = 1.0, alpha: float = 0.5, 
         record_iterates: bool = True):
    """
    Run grid feedback optimizer from a JSON file path provided as string.
    """
    # Convert to Path
    json_path = Path(json_path)

    # Load network
    network = load_network(json_path)

    # Solve
    results = solve(network, max_iter = max_iter, tol = tol, 
                                       delta_p = delta_p, delta_q = delta_q,
                                       alpha = alpha, record_iterates = record_iterates)
    

    return results



