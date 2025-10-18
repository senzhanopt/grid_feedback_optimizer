from pathlib import Path
from grid_feedback_optimizer.models.loader import load_network
from grid_feedback_optimizer.engine.solve import solve


def main(json_path: str, max_iter: int = 1000, tol: float = 1e-3,
          delta_p: float = 1.0, delta_q: float = 1.0, algorithm: str = "gp", 
          alpha: float = 0.5, alpha_v: float = 10.0, 
          alpha_l: float = 10.0, alpha_t: float = 10.0, record_iterates: bool = True,
          solver: str = "CLARABLE", loading_meas_side: str = "from"):
    """
    Run grid feedback optimizer from a JSON file path provided as string.
    """
    # Convert to Path
    json_path = Path(json_path)

    # Load network
    network = load_network(json_path)

    # Solve
    results = solve(network, max_iter = max_iter, tol = tol, delta_p = delta_p, delta_q = delta_q,
                    algorithm = algorithm, alpha = alpha, alpha_v = alpha_v, alpha_l = alpha_l,
                    alpha_t = alpha_t, record_iterates = record_iterates, solver = solver,
                    loading_meas_side = loading_meas_side)
    

    return results



