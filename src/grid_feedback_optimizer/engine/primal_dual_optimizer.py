from grid_feedback_optimizer.models.network import Network
from grid_feedback_optimizer.engine.renew_gen_projection import RenewGenProjection
from grid_feedback_optimizer.models.solve_data import OptimizationInputs
import numpy as np
import math
from collections.abc import Callable

class PrimalDualOptimizer:
    """
    A primal-dual gradient projection feedback optimizer.
    """
    def __init__(self, network: Network, sensitivities: dict, alpha: float = 0.5,
                 alpha_v: float = 10.0, alpha_l: float = 10.0, alpha_t: float = 10.0,
                 solver: str = "CLARABLE"):
        """
        Initialize optimizer and build cached problem.
        """
        self.sensitivities = sensitivities
        self.alpha = alpha
        self.alpha_v = alpha_v
        self.alpha_l = alpha_l
        self.alpha_t = alpha_t
        self.renew_gen_proj = RenewGenProjection(solver = solver)

        # === Scaling factors ===
        s_inv_mean = np.mean([gen.s_inv for gen in network.renew_gens])
        self.param_scale = 10**math.floor(math.log10(s_inv_mean)) # Round down to nearest lower power of 10

        # read parameters from network
        self.n_bus = len(network.buses)
        self.n_line = len(network.lines)
        self.n_transformer = len(network.transformers)
        self.n_gen = len(network.renew_gens)

        self.u_pu_max = np.array([bus.u_pu_max for bus in network.buses])
        self.u_pu_min = np.array([bus.u_pu_min for bus in network.buses])
        self.s_line = np.sqrt(3) * np.array([line.i_n * network.buses[line.from_bus].u_rated for line in network.lines])

        self.c2_p = np.array([gen.c2_p for gen in network.renew_gens])
        self.c1_p = np.array([gen.c1_p for gen in network.renew_gens])
        self.c2_q = np.array([gen.c2_q for gen in network.renew_gens])
        self.c1_q = np.array([gen.c1_q for gen in network.renew_gens])
        self.p_norm = np.array([gen.p_norm for gen in network.renew_gens])/self.param_scale
        self.q_norm = np.array([gen.q_norm for gen in network.renew_gens])/self.param_scale
        self.p_min = np.array([gen.p_min for gen in network.renew_gens])/self.param_scale
        self.p_max = np.array([gen.p_max for gen in network.renew_gens])/self.param_scale
        self.s_inv = np.array([gen.s_inv for gen in network.renew_gens])/self.param_scale
        
        # initialize dual variables
        self.dual_v_upp = np.zeros(self.n_bus)        
        self.dual_v_low = np.zeros(self.n_bus)        
        self.dual_line = np.zeros(self.n_line)        
        if self.n_transformer >= 1:
            self.dual_transformer = np.zeros(self.n_transformer)
            self.s_transformer = np.array([transformer.sn for transformer in network.transformers])

    @staticmethod
    def calc_loading(p: np.ndarray | float, q: np.ndarray | float, s: np.ndarray | float) -> np.ndarray | float:
        return np.sqrt(p**2 + q**2) / s
    
    @staticmethod
    def calc_pf(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray | float:
        return p / np.sqrt(p**2 + q**2)

    @staticmethod
    def calc_rpf(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray | float:
        return q / np.sqrt(p**2 + q**2)


    def solve_problem(self, opt_input: OptimizationInputs, grad_callback: Callable | None = None, **callback_kwargs):
        """
        Update parameters and implement primal-dual gradient projection.

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
        grad_callback: callable | None = None
            A function which takes grad_p and grad_q and outputs grad_p and grad_q

        Returns
        -------
        np.ndarray
            Optimized generator setpoints of shape (n_generators, 2),
            where each row contains [p_opt, q_opt].
        """

        # update dual variables
        param_dict = opt_input.to_dict()
        self.dual_v_upp += self.alpha_v * (param_dict["u_pu_meas"] - self.u_pu_max)
        self.dual_v_upp[self.dual_v_upp < 0] = 0.0
        self.dual_v_low += self.alpha_v * (self.u_pu_min - param_dict["u_pu_meas"])
        self.dual_v_low[self.dual_v_low < 0] = 0.0
        self.dual_line += self.alpha_l * (self.calc_loading(param_dict["P_line_meas"], param_dict["Q_line_meas"], self.s_line)-1.0)
        self.dual_line[self.dual_line < 0] = 0.0
        if self.n_transformer >= 1:
            self.dual_transformer += self.alpha_t * (self.calc_loading(param_dict["P_transformer_meas"], param_dict["Q_transformer_meas"], self.s_transformer)-1.0)
            self.dual_transformer[self.dual_transformer < 0] = 0.0

        # update primal variables
        grad_p = 2.0 * self.c2_p * (param_dict["p_gen_last"]/self.param_scale - self.p_norm) + self.c1_p
        grad_p += self.sensitivities["du_dp"].T @ (self.dual_v_upp - self.dual_v_low)*self.param_scale
        grad_p += self.sensitivities["dP_line_dp"].T @ (self.dual_line * self.calc_pf(param_dict["P_line_meas"],param_dict["Q_line_meas"]) / self.s_line)*self.param_scale
        grad_p += self.sensitivities["dQ_line_dp"].T @ (self.dual_line * self.calc_rpf(param_dict["P_line_meas"],param_dict["Q_line_meas"]) / self.s_line)*self.param_scale
        if self.n_transformer >= 1:
            grad_p += self.sensitivities["dP_transformer_dp"].T @ (self.dual_transformer * self.calc_pf(param_dict["P_transformer_meas"],param_dict["Q_transformer_meas"]) / self.s_transformer)*self.param_scale
            grad_p += self.sensitivities["dQ_transformer_dp"].T @ (self.dual_transformer * self.calc_rpf(param_dict["P_transformer_meas"],param_dict["Q_transformer_meas"]) / self.s_transformer)*self.param_scale

        grad_q = 2.0 * self.c2_q * (param_dict["q_gen_last"]/self.param_scale - self.q_norm) + self.c1_q
        grad_q += self.sensitivities["du_dq"].T @ (self.dual_v_upp - self.dual_v_low)*self.param_scale
        grad_q += self.sensitivities["dP_line_dq"].T @ (self.dual_line * self.calc_pf(param_dict["P_line_meas"],param_dict["Q_line_meas"]) / self.s_line)*self.param_scale
        grad_q += self.sensitivities["dQ_line_dq"].T @ (self.dual_line * self.calc_rpf(param_dict["P_line_meas"],param_dict["Q_line_meas"]) / self.s_line)*self.param_scale
        if self.n_transformer >= 1:
            grad_q += self.sensitivities["dP_transformer_dq"].T @ (self.dual_transformer * self.calc_pf(param_dict["P_transformer_meas"],param_dict["Q_transformer_meas"]) / self.s_transformer)*self.param_scale
            grad_q += self.sensitivities["dQ_transformer_dq"].T @ (self.dual_transformer * self.calc_rpf(param_dict["P_transformer_meas"],param_dict["Q_transformer_meas"]) / self.s_transformer)*self.param_scale

        # apply callback function, e.g. when including an extra gradient term from transformer-level setpoint tracking
        if grad_callback is not None:
            grad_p, grad_q = grad_callback(grad_p, grad_q, **callback_kwargs)

        # gradient descent
        p_target = param_dict["p_gen_last"]/self.param_scale - self.alpha * grad_p
        q_target = param_dict["q_gen_last"]/self.param_scale - self.alpha * grad_q

        # call projection
        p_gen_opt = np.zeros(self.n_gen)
        q_gen_opt = np.zeros(self.n_gen)
        for i in range(self.n_gen):
            if self.p_min[i] == 0.0: # generator
                p_gen_opt[i], q_gen_opt[i] = self.renew_gen_proj.analytic_projection(self.p_max[i], self.s_inv[i], p_target[i], q_target[i])
            elif self.p_max[i] == 0.0: # flex load
                p_gen_opt[i], q_gen_opt[i] = self.renew_gen_proj.analytic_projection(-self.p_min[i], self.s_inv[i], -p_target[i], q_target[i])
                p_gen_opt[i] *= -1.0
            elif self.p_min[i] < 0 and self.p_max[i] > 0: # flex gen/load
                if p_target[i] >= 0:
                    p_gen_opt[i], q_gen_opt[i] = self.renew_gen_proj.analytic_projection(self.p_max[i], self.s_inv[i], p_target[i], q_target[i])
                else:
                    p_gen_opt[i], q_gen_opt[i] = self.renew_gen_proj.analytic_projection(-self.p_min[i], self.s_inv[i], -p_target[i], q_target[i])
                    p_gen_opt[i] *= -1.0
            else: # no easy analytic solution
                p_gen_opt[i], q_gen_opt[i] = self.renew_gen_proj.opt_projection(self.p_max[i], self.p_min[i], self.s_inv[i], p_target[i], q_target[i])
        
        return self.param_scale * np.column_stack((p_gen_opt, q_gen_opt))