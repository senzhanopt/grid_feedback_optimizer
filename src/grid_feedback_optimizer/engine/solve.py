import numpy as np
from grid_feedback_optimizer.models.network import Network
from grid_feedback_optimizer.engine.powerflow import PowerFlowSolver
from grid_feedback_optimizer.engine.optimization import GradientProjectionOptimizer
import copy
from power_grid_model import ComponentType
from grid_feedback_optimizer.models.solve_data import SolveResults

def solve(network: Network, max_iter: int = 100, tol: float = 1e-4,
          delta_p: float = 1.0, delta_q: float = 1.0, alpha: float = 0.5, 
          record_iterates: bool = True):
    """
    Solve the grid optimization problem by iterating
    between power flow and optimization.
    """
    # Initialize solver and optimizer
    n_transformer = len(network.transformers)
    power_flow_solver = PowerFlowSolver(network)
    sensitivities = power_flow_solver.obtain_sensitivity(delta_p = delta_p, delta_q = delta_q)
    optimizer = GradientProjectionOptimizer(network, sensitivities, alpha = alpha)

    # Iterative loop
    output_data = copy.deepcopy(power_flow_solver.base_output_data)
    gen_update = np.column_stack((power_flow_solver.base_p_gen,power_flow_solver.base_q_gen))
    iterates = []

    for k in range(1,max_iter+1):
        # 1. Get current network state
        u_pu_meas = np.array(output_data[ComponentType.node]["u_pu"])
        P_line_meas = np.array(output_data[ComponentType.line]["p_from"])
        Q_line_meas = np.array(output_data[ComponentType.line]["q_from"])
        if n_transformer >= 1:
            P_transformer_meas = np.array(output_data[ComponentType.transformer]["p_from"])
            Q_transformer_meas = np.array(output_data[ComponentType.transformer]["q_from"])
        # 2. Run optimization step → propose new setpoints
        param_dict = {
            "u_pu_meas": u_pu_meas,
            "P_line_meas": P_line_meas,
            "Q_line_meas": Q_line_meas,
            "p_gen_last": gen_update[:,0],
            "q_gen_last": gen_update[:,1],
        }
        if n_transformer >= 1:
            param_dict["P_transformer_meas"] = P_transformer_meas
            param_dict["Q_transformer_meas"] = Q_transformer_meas

        gen_update = optimizer.solve_problem(param_dict)

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