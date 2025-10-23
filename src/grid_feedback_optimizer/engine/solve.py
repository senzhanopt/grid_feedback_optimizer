import numpy as np
from grid_feedback_optimizer.models.network import Network
from grid_feedback_optimizer.engine.powerflow import PowerFlowSolver
from grid_feedback_optimizer.engine.grad_proj_optimizer import GradientProjectionOptimizer
from grid_feedback_optimizer.engine.primal_dual_optimizer import PrimalDualOptimizer
import copy
from power_grid_model import ComponentType
from grid_feedback_optimizer.models.solve_data import SolveResults, OptimizationInputs

def solve(network: Network, max_iter: int = 1000, tol: float = 1e-3,
          delta_p: float = 1.0, delta_q: float = 1.0, algorithm: str = "gp", 
          alpha: float = 0.5, alpha_v: float = 10.0, 
          alpha_l: float = 10.0, alpha_t: float = 10.0, record_iterates: bool = True,
          solver: str = "CLARABEL", loading_meas_side: str = "from",
          rel_tol = 1E-4, rel_tol_line = 1E-2, **solver_kwargs):
    """
    Solve the grid optimization problem by iterating
    between power flow and optimization.
    """
    # Initialize solver and optimizer
    n_transformer = len(network.transformers)
    power_flow_solver = PowerFlowSolver(network)
    sensitivities = power_flow_solver.obtain_sensitivity(delta_p = delta_p, delta_q = delta_q, loading_meas_side = loading_meas_side,
                                                         rel_tol = rel_tol, rel_tol_line = rel_tol_line)
    
    if algorithm == "gp":
        optimizer = GradientProjectionOptimizer(network, sensitivities, alpha = alpha, solver = solver, **solver_kwargs)
    elif algorithm == "pd":
        optimizer = PrimalDualOptimizer(network, sensitivities, alpha = alpha,
                                                alpha_v = alpha_v, alpha_l = alpha_l, alpha_t = alpha_t, solver = solver, **solver_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")    

    # Iterative loop
    output_data = copy.deepcopy(power_flow_solver.base_output_data)
    gen_update = np.column_stack((power_flow_solver.base_p_gen,power_flow_solver.base_q_gen))
    iterates = []

    if record_iterates:
        iterates.append({
            "iteration": 0,
            "gen_update": gen_update.copy(),
            "output_data": copy.deepcopy(output_data)
        })

    for k in range(1,max_iter+1):
        # 1. Get current network state
        u_pu_meas = np.array(output_data[ComponentType.node]["u_pu"])
        P_line_meas = np.array(output_data[ComponentType.line]["p_"+loading_meas_side])
        Q_line_meas = np.array(output_data[ComponentType.line]["q_"+loading_meas_side])
        if n_transformer >= 1:
            P_transformer_meas = np.array(output_data[ComponentType.transformer]["p_"+loading_meas_side])
            Q_transformer_meas = np.array(output_data[ComponentType.transformer]["q_"+loading_meas_side])
        # 2. Run optimization step → propose new setpoints
        param_dict = {
            "u_pu_meas": u_pu_meas,
            "P_line_meas": P_line_meas,
            "Q_line_meas": Q_line_meas,
            "p_gen_last": gen_update[:, 0],
            "q_gen_last": gen_update[:, 1],
        }

        if n_transformer >= 1:
            param_dict.update({
                "P_transformer_meas": P_transformer_meas,
                "Q_transformer_meas": Q_transformer_meas
            })

        opt_input = OptimizationInputs(**param_dict)

        gen_update = optimizer.solve_problem(opt_input)

        # 3. Run power flow with updated setpoints
        output_data = power_flow_solver.run(gen_update=gen_update)

        if record_iterates:
            iterates.append({
                "iteration": k,
                "gen_update": gen_update.copy(),
                "output_data": copy.deepcopy(output_data)
            })

        # 4. Check convergence
        if np.max(np.abs(gen_update-np.column_stack((param_dict["p_gen_last"], param_dict["q_gen_last"])))) < tol:
            print("Converged ✅")
            break
    
        
    return SolveResults(final_output=output_data, final_gen_update=gen_update, iterations=iterates)