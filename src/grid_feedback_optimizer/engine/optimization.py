import cvxpy as cp
import numpy as np
from grid_feedback_optimizer.models.network import Network

class GradientProjectionOptimizer:
    """
    Gradient projection optimizer.
    Caches the CVXPY problem to allow fast updates of parameters.
    """

    def __init__(self, network: Network, sensitivities: dict, alpha: float = 0.5):
        """
        Initialize optimizer and build cached problem.
        """
        self.prob = self.build_problem(network, sensitivities, alpha)


    def build_problem(self, network: Network, sensitivities: dict, alpha: float, c_q = 0.1):
        """
        Build CVXPY problem with Parameters and Variables.
        Only called once. Returns dictionary with problem and variables/parameters.
        """    
        n_bus = len(network.buses)
        n_line = len(network.lines)
        self.n_transformer = len(network.transformers)
        n_gen = len(network.renew_gens)

        # Parameters  
        self.u_pu_meas = cp.Parameter(n_bus)
        self.P_line_meas = cp.Parameter(n_line)
        self.Q_line_meas = cp.Parameter(n_line)
        if self.n_transformer >= 1:
            self.P_transformer_meas = cp.Parameter(self.n_transformer)
            self.Q_transformer_meas = cp.Parameter(self.n_transformer)
        self.p_gen_last = cp.Parameter(n_gen)
        self.q_gen_last = cp.Parameter(n_gen)

        # Variables
        self.p_gen = cp.Variable(n_gen)
        self.q_gen = cp.Variable(n_gen)

        # Constraints
        cons = []
        
        # renewable gens
        for i in range(n_gen):
            cons += [self.p_gen[i] <= network.renew_gens[i].p_max]
            cons += [self.p_gen[i] >= network.renew_gens[i].p_min]
            cons += [cp.SOC(network.renew_gens[i].s_inv, cp.hstack([self.p_gen[i], self.q_gen[i]]))]
        
        # voltage
        cons += [self.u_pu_meas + sensitivities["du_dp"]@(self.p_gen - self.p_gen_last)
                 + sensitivities["du_dq"]@(self.q_gen - self.q_gen_last) <= np.array([bus.u_pu_max for bus in network.buses])]
        cons += [self.u_pu_meas + sensitivities["du_dp"]@(self.p_gen - self.p_gen_last)
                 + sensitivities["du_dq"]@(self.q_gen - self.q_gen_last) >= np.array([bus.u_pu_min for bus in network.buses])]
        
        # line
        for l in range(n_line):
            line = network.lines[l]
            s_line = np.sqrt(3) * line.i_n * network.buses[line.from_bus].u_rated
            cons += [cp.SOC(s_line, cp.hstack([
                self.P_line_meas[l] + sensitivities["dP_line_dp"][l,:]@(self.p_gen - self.p_gen_last) + sensitivities["dP_line_dq"][l,:]@(self.q_gen - self.q_gen_last),
                self.Q_line_meas[l] + sensitivities["dQ_line_dp"][l,:]@(self.p_gen - self.p_gen_last) + sensitivities["dQ_line_dq"][l,:]@(self.q_gen - self.q_gen_last)
            ]))]    

        # transformer
        if self.n_transformer >= 1:
            for t in range(self.n_transformer):
                transformer = network.transformers[t]
                s_transformer = transformer.sn
                cons += [cp.SOC(s_transformer, cp.hstack([
                    self.P_transformer_meas[t] + sensitivities["dP_transformer_dp"][t,:]@(self.p_gen - self.p_gen_last) + sensitivities["dP_transformer_dq"][t,:]@(self.q_gen - self.q_gen_last),
                    self.Q_transformer_meas[t] + sensitivities["dQ_transformer_dp"][t,:]@(self.p_gen - self.p_gen_last) + sensitivities["dQ_transformer_dq"][t,:]@(self.q_gen - self.q_gen_last)
                ]))]                 
        
        # Objective
        grad_p = self.p_gen_last - np.array([gen.p_max for gen in network.renew_gens])
        grad_q = c_q * self.q_gen_last
        
        objective = cp.Minimize(
            cp.sum_squares(self.p_gen - (self.p_gen_last - alpha * grad_p)) +
            cp.sum_squares(self.q_gen - (self.q_gen_last - alpha * grad_q))
        )

        # construct problem
        prob = cp.Problem(objective, cons)

        return prob

    def solve_problem(self, param_dict: dict):
        """
        Update parameters and solve the cached optimization problem.

        Parameters
        ----------
        param_dict : dict
            Must contain keys: "u_pu_meas", "P_line_meas", "Q_line_meas",
            "p_gen_last", "q_gen_last"
        
        Returns
        -------
        p_gen_opt, q_gen_opt : np.ndarray
            Optimized active and reactive power setpoints for generators
        """
        # Update CVXPY parameter values
        self.u_pu_meas.value = param_dict["u_pu_meas"]
        self.P_line_meas.value = param_dict["P_line_meas"]
        self.Q_line_meas.value = param_dict["Q_line_meas"]
        self.p_gen_last.value = param_dict["p_gen_last"]
        self.q_gen_last.value = param_dict["q_gen_last"]
        if self.n_transformer >= 1:
            self.P_transformer_meas.value = param_dict["P_transformer_meas"]
            self.Q_transformer_meas.value = param_dict["Q_transformer_meas"]
        

        try:
            self.prob.solve()
        except:
            print(self.prob.status)
            # If solver fails, return last known feasible values
            print("Solver failed, returning previous generator setpoints.")
            return np.column_stack((param_dict["p_gen_last"], param_dict["q_gen_last"]))
        else:
            if  self.prob.status == "optimal":
                return np.column_stack((self.p_gen.value, self.q_gen.value))
            else:            
                return np.column_stack((param_dict["p_gen_last"], param_dict["q_gen_last"]))
