import argparse
from pathlib import Path

from grid_feedback_optimizer.engine.solve import solve
from grid_feedback_optimizer.models.loader import load_network, load_network_from_excel


def main(
    file_path: str,
    save_path: str | None = None,
    max_iter: int = 1000,
    tol: float = 1e-3,
    delta_p: float = 1.0,
    delta_q: float = 1.0,
    algorithm: str = "gp",
    alpha: float = 0.5,
    alpha_v: float = 10.0,
    alpha_l: float = 10.0,
    alpha_t: float = 10.0,
    no_record_iterates: bool = False,
    solver: str = "CLARABEL",
    loading_meas_side: str = "from",
    rel_tol=1e-4,
    rel_tol_line=1e-2,
    **solver_kwargs,
):
    """
    Run grid feedback optimizer from a JSON/EXCEL file path provided as string.
    """
    # Convert to Path
    file_path = Path(file_path)

    # Load network
    if file_path.suffix == ".xlsx":
        network = load_network_from_excel(file_path)
    else:
        network = load_network(file_path)

    # Solve
    results = solve(
        network,
        max_iter=max_iter,
        tol=tol,
        delta_p=delta_p,
        delta_q=delta_q,
        algorithm=algorithm,
        alpha=alpha,
        alpha_v=alpha_v,
        alpha_l=alpha_l,
        alpha_t=alpha_t,
        record_iterates=not no_record_iterates,
        solver=solver,
        loading_meas_side=loading_meas_side,
        rel_tol=rel_tol,
        rel_tol_line=rel_tol_line,
        **solver_kwargs,
    )

    results.print_summary()

    if save_path:
        results.save(save_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Grid Feedback Optimizer")
    parser.add_argument(
        "file_path", type=str, help="Path to network JSON or Excel file"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save results"
    )
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--delta_p", type=float, default=1.0)
    parser.add_argument("--delta_q", type=float, default=1.0)
    parser.add_argument("--algorithm", type=str, default="gp")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--alpha_v", type=float, default=10.0)
    parser.add_argument("--alpha_l", type=float, default=10.0)
    parser.add_argument("--alpha_t", type=float, default=10.0)
    parser.add_argument("--no_record_iterates", action="store_false")
    parser.add_argument("--solver", type=str, default="CLARABEL")
    parser.add_argument("--loading_meas_side", type=str, default="from")
    parser.add_argument("--rel_tol", type=float, default=1e-4)
    parser.add_argument("--rel_tol_line", type=float, default=1e-2)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(**vars(args))
