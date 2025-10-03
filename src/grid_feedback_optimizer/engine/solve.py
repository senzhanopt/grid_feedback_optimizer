import numpy as np
from grid_feedback_optimizer.models.network import Network
from grid_feedback_optimizer.engine.powerflow import PowerFlowSolver
from grid_feedback_optimizer.engine.optimization import GradientProjectionOptimizer
from grid_feedback_optimizer.utils.utils import print_component
import copy
from power_grid_model import ComponentType

def solve(network: Network, max_iter: int = 20, tol: float = 1e-4):
    """
    Solve the grid optimization problem by iterating
    between power flow and optimization.
    """
    # Initialize solver and optimizer
    power_flow_solver = PowerFlowSolver(network)
    sensitivities = power_flow_solver.obtain_sensitivity()
    optimizer = GradientProjectionOptimizer(network, sensitivities)

    # Base case
    print("==== Iteration 0 (Base) ====")
    print_component(power_flow_solver.base_output_data, "node")
    print_component(power_flow_solver.base_output_data, "line")


    # Iterative loop
    output_data = copy.deepcopy(power_flow_solver.base_output_data)
    p_gen_itr = copy.deepcopy(power_flow_solver.base_p_gen)
    q_gen_itr = copy.deepcopy(power_flow_solver.base_q_gen)
    for k in range(max_iter):
        # 1. Get current network state
        u_pu_meas = np.array(output_data[ComponentType.node]["u_pu"])
        P_line_meas = np.array(output_data[ComponentType.line]["p_from"])
        Q_line_meas = np.array(output_data[ComponentType.line]["q_from"])

        # 2. Run optimization step → propose new setpoints
        param_dict = {
            "u_pu_meas": u_pu_meas,
            "P_line_meas": P_line_meas,
            "Q_line_meas": Q_line_meas,
            "p_gen_last": p_gen_itr,
            "q_gen_last": q_gen_itr,
        }
        gen_update = optimizer.solve_problem(param_dict)

        # 3. Run power flow with updated setpoints
        output_data = power_flow_solver.run(gen_update=gen_update)

        # 4. Check convergence
        if np.max(np.abs(gen_update)) < tol:
            print("Converged ✅")
            break
        
    
    return output_data