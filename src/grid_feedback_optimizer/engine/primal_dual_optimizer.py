from grid_feedback_optimizer.engine.renew_gen_projection import RenewGenProjection
from grid_feedback_optimizer.models.solve_data import OptimizationInputs, OptimizationModelData
import numpy as np
import math
from collections.abc import Callable

class PrimalDualOptimizer:
    """
    A primal-dual gradient projection feedback optimizer.
    """
    def __init__(self, opt_model_data: OptimizationModelData, sensitivities: dict, alpha: float = 0.5,
                 alpha_v: float = 10.0, alpha_l: float = 10.0, alpha_t: float = 10.0,
                 solver: str = "CLARABEL", **solver_kwargs):
        """
        Initialize optimizer and build cached problem.
        """
        self.sensitivities = sensitivities
        self.alpha = alpha
        self.alpha_v = alpha_v
        self.alpha_l = alpha_l
        self.alpha_t = alpha_t
        self.renew_gen_proj = RenewGenProjection(solver = solver, **solver_kwargs)

        # read parameters from opt_model_data
        self.n_bus = len(opt_model_data.u_pu_max)
        self.n_line = len(opt_model_data.s_line)
        self.n_transformer = len(opt_model_data.s_transformer)
        self.n_gen = len(opt_model_data.p_min)

        # === Scaling factors ===
        p_abs_mean = np.mean([max(abs(opt_model_data.p_max[i]), abs(opt_model_data.p_min[i])) for i in range(self.n_gen)])
        self.param_scale = 10**math.floor(math.log10(p_abs_mean)) # Round down to nearest lower power of 10

        self.u_pu_max = opt_model_data.u_pu_max
        self.u_pu_min = opt_model_data.u_pu_min
        self.s_line = opt_model_data.s_line

        self.c2_p = opt_model_data.c2_p
        self.c1_p = opt_model_data.c1_p
        self.c2_q = opt_model_data.c2_q
        self.c1_q = opt_model_data.c1_q
        self.p_norm = opt_model_data.p_norm
        self.q_norm = opt_model_data.q_norm
        self.p_min = opt_model_data.p_min
        self.p_max = opt_model_data.p_max

        self.s_inv = opt_model_data.s_inv
        self.pf_min = opt_model_data.pf_min
        self.q_min = opt_model_data.q_min
        self.q_max = opt_model_data.q_max
        
        # initialize dual variables
        self.dual_v_upp = np.zeros(self.n_bus)        
        self.dual_v_low = np.zeros(self.n_bus)        
        self.dual_line = np.zeros(self.n_line)        
        if self.n_transformer >= 1:
            self.dual_transformer = np.zeros(self.n_transformer)
            self.s_transformer = opt_model_data.s_transformer

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
        grad_p = 2.0 * self.c2_p * (param_dict["p_gen_last"]/self.param_scale - self.p_norm/self.param_scale) + self.c1_p
        grad_p += self.sensitivities["du_dp"].T @ (self.dual_v_upp - self.dual_v_low)*self.param_scale
        grad_p += self.sensitivities["dP_line_dp"].T @ (self.dual_line * self.calc_pf(param_dict["P_line_meas"],param_dict["Q_line_meas"]) / self.s_line)*self.param_scale
        grad_p += self.sensitivities["dQ_line_dp"].T @ (self.dual_line * self.calc_rpf(param_dict["P_line_meas"],param_dict["Q_line_meas"]) / self.s_line)*self.param_scale
        if self.n_transformer >= 1:
            grad_p += self.sensitivities["dP_transformer_dp"].T @ (self.dual_transformer * self.calc_pf(param_dict["P_transformer_meas"],param_dict["Q_transformer_meas"]) / self.s_transformer)*self.param_scale
            grad_p += self.sensitivities["dQ_transformer_dp"].T @ (self.dual_transformer * self.calc_rpf(param_dict["P_transformer_meas"],param_dict["Q_transformer_meas"]) / self.s_transformer)*self.param_scale

        grad_q = 2.0 * self.c2_q * (param_dict["q_gen_last"]/self.param_scale - self.q_norm/self.param_scale) + self.c1_q
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

        # expose them for advanced use of this library
        self.p_target = p_target * self.param_scale
        self.q_target = q_target * self.param_scale

        # call projection
        p_gen_opt = np.zeros(self.n_gen)
        q_gen_opt = np.zeros(self.n_gen)
        for i in range(self.n_gen):
            p_gen_opt[i], q_gen_opt[i] = self.renew_gen_proj.projection(p_max = self.p_max[i]/self.param_scale,
                                                                        p_min = self.p_min[i]/self.param_scale, 
                                                                        p = p_target[i], 
                                                                        q = q_target[i],
                                                                        s_inv = self.s_inv[i]/self.param_scale if self.s_inv[i] is not None else None,
                                                                        pf_min = self.pf_min[i],
                                                                        q_min = self.q_min[i]/self.param_scale if self.q_min[i] is not None else None,
                                                                        q_max = self.q_max[i]/self.param_scale if self.q_max[i] is not None else None)
        
        return self.param_scale * np.column_stack((p_gen_opt, q_gen_opt))