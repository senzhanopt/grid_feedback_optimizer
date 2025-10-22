import cvxpy as cp
import numpy as np
import math
from grid_feedback_optimizer.models.network import Network
from grid_feedback_optimizer.models.solve_data import OptimizationInputs

class GradientProjectionOptimizer:
    """
    Gradient projection optimizer.
    Caches the CVXPY problem to allow fast updates of parameters.
    """

    def __init__(self, network: Network, sensitivities: dict, alpha: float = 0.5, solver: str = "CLARABEL"):
        """
        Initialize optimizer and build cached problem.
        """
        self.prob, self.cons, self.obj = self._build_problem(network, sensitivities, alpha)
        self.solver = solver


    def _build_problem(self, network: Network, sensitivities: dict, alpha: float):
        """
        Build CVXPY problem with Parameters and Variables.
        Only called once. Returns dictionary with problem and variables/parameters.
        """    
        n_bus = len(network.buses)
        n_line = len(network.lines)
        self.n_transformer = len(network.transformers)
        n_gen = len(network.renew_gens)

        # === Scaling factors ===
        s_inv_mean = np.mean([gen.s_inv for gen in network.renew_gens])
        self.param_scale = 10**math.floor(math.log10(s_inv_mean)) # Round down to nearest lower power of 10

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
        # scaled value instead of values in Watt
        self.p_gen = cp.Variable(n_gen)
        self.q_gen = cp.Variable(n_gen)

        # Parameters with given values
        self.p_max = cp.Parameter(n_gen)
        self.p_min = cp.Parameter(n_gen)
        self.p_norm = cp.Parameter(n_gen)
        self.q_norm = cp.Parameter(n_gen)

        self.p_max.value = np.array([g.p_max for g in network.renew_gens])
        self.p_min.value = np.array([g.p_min for g in network.renew_gens])
        self.p_norm.value = np.array([g.p_norm for g in network.renew_gens])
        self.q_norm.value = np.array([g.q_norm for g in network.renew_gens])

        # Constraints
        cons = []
        
        # renewable gens
        for i in range(n_gen):
            cons += [self.p_gen[i] <= self.p_max[i] / self.param_scale]
            cons += [self.p_gen[i] >= self.p_min[i] / self.param_scale]
            cons += [cp.SOC(1.0, cp.hstack([self.p_gen[i], self.q_gen[i]])/(network.renew_gens[i].s_inv / self.param_scale))]
        
        # voltage
        cons += [self.u_pu_meas + sensitivities["du_dp"]@(self.p_gen * self.param_scale - self.p_gen_last) 
                 + sensitivities["du_dq"]@(self.q_gen * self.param_scale  - self.q_gen_last) <= np.array([bus.u_pu_max for bus in network.buses])]
        cons += [self.u_pu_meas + sensitivities["du_dp"]@(self.p_gen * self.param_scale - self.p_gen_last)
                 + sensitivities["du_dq"]@(self.q_gen  * self.param_scale - self.q_gen_last) >= np.array([bus.u_pu_min for bus in network.buses])]
        
        # line
        for l in range(n_line):
            line = network.lines[l]
            s_line = np.sqrt(3) * line.i_n * network.buses[line.from_bus].u_rated
            cons += [cp.SOC(1.0, cp.hstack([
                self.P_line_meas[l] + sensitivities["dP_line_dp"][l,:]@(self.p_gen * self.param_scale - self.p_gen_last) + sensitivities["dP_line_dq"][l,:]@(self.q_gen * self.param_scale - self.q_gen_last),
                self.Q_line_meas[l] + sensitivities["dQ_line_dp"][l,:]@(self.p_gen * self.param_scale - self.p_gen_last) + sensitivities["dQ_line_dq"][l,:]@(self.q_gen * self.param_scale - self.q_gen_last)
            ])/s_line)]    

        # transformer
        if self.n_transformer >= 1:
            for t in range(self.n_transformer):
                transformer = network.transformers[t]
                s_transformer = transformer.sn
                cons += [cp.SOC(1.0, cp.hstack([
                    self.P_transformer_meas[t] + sensitivities["dP_transformer_dp"][t,:]@(self.p_gen * self.param_scale - self.p_gen_last) + sensitivities["dP_transformer_dq"][t,:]@(self.q_gen * self.param_scale - self.q_gen_last),
                    self.Q_transformer_meas[t] + sensitivities["dQ_transformer_dp"][t,:]@(self.p_gen * self.param_scale - self.p_gen_last) + sensitivities["dQ_transformer_dq"][t,:]@(self.q_gen * self.param_scale - self.q_gen_last)
                ])/s_transformer)]                 
        
        # Objective
        grad_p = 2.0 * cp.multiply( np.array([gen.c2_p for gen in network.renew_gens]), (self.p_gen_last - self.p_norm))
        grad_p += np.array([gen.c1_p for gen in network.renew_gens])
        grad_q = 2.0 * cp.multiply(np.array([gen.c2_q for gen in network.renew_gens]), (self.q_gen_last - self.q_norm))
        grad_q += np.array([gen.c1_q for gen in network.renew_gens])

        obj = cp.Minimize(
            cp.sum_squares(self.p_gen - (self.p_gen_last - alpha * grad_p) / self.param_scale) +
            cp.sum_squares(self.q_gen - (self.q_gen_last - alpha * grad_q) / self.param_scale)
        )

        # construct problem
        prob = cp.Problem(obj, cons)

        return prob, cons, obj

    def solve_problem(self, opt_input: OptimizationInputs):
        """
        Update CVXPY parameters using structured optimization inputs,
        solve the cached optimization problem, and return optimized setpoints.

        Parameters
        ----------
        opt_input : OptimizationInputs
            Structured input model containing:
                - u_pu_meas : np.ndarray
                    Measured node voltages [p.u.]
                - P_line_meas, Q_line_meas : np.ndarray
                    Measured active/reactive line power flows
                - p_gen_last, q_gen_last : np.ndarray
                    Previous generator active/reactive power setpoints
                - P_transformer_meas, Q_transformer_meas : np.ndarray, optional
                    Measured transformer active/reactive power (if applicable)

        Returns
        -------
        np.ndarray
            Optimized generator setpoints of shape (n_generators, 2),
            where each row contains [p_opt, q_opt].
        """

        # Update CVXPY parameter values
        param_dict = opt_input.to_dict()
        self.u_pu_meas.value = param_dict["u_pu_meas"]
        self.P_line_meas.value = param_dict["P_line_meas"]
        self.Q_line_meas.value = param_dict["Q_line_meas"]
        self.p_gen_last.value = param_dict["p_gen_last"]
        self.q_gen_last.value = param_dict["q_gen_last"]
        if self.n_transformer >= 1:
            self.P_transformer_meas.value = param_dict["P_transformer_meas"]
            self.Q_transformer_meas.value = param_dict["Q_transformer_meas"]
        

        try:
            self.prob.solve(solver = getattr(cp, self.solver))
        except:
            print(self.prob.status)
            # If solver fails, return last known feasible values
            print("Solver failed, returning previous generator setpoints.")
            return np.column_stack((param_dict["p_gen_last"], param_dict["q_gen_last"]))
        else:
            if  self.prob.status == "optimal":
                return np.column_stack((self.p_gen.value, self.q_gen.value)) * self.param_scale
            else:            
                return np.column_stack((param_dict["p_gen_last"], param_dict["q_gen_last"]))
