import math

import cvxpy as cp
import numpy as np

from grid_feedback_optimizer.models.solve_data import (
    OptimizationInputs,
    OptimizationModelData,
)


class GradientProjectionOptimizer:
    """
    Gradient projection optimizer.
    Caches the CVXPY problem to allow fast updates of parameters.
    """

    def __init__(
        self,
        opt_model_data: OptimizationModelData,
        sensitivities: dict,
        alpha: float = 0.5,
        solver: str = "CLARABEL",
        **solver_kwargs,
    ):
        """
        Initialize optimizer and build cached problem.
        """
        self.prob, self.cons, self.obj = self._build_problem(
            opt_model_data, sensitivities, alpha
        )
        self.solver = solver
        self.solver_kwargs = solver_kwargs

    def _build_problem(
        self, opt_model_data: OptimizationModelData, sensitivities: dict, alpha: float
    ):
        """
        Build CVXPY problem with Parameters and Variables.
        Only called once. Returns dictionary with problem and variables/parameters.
        """
        n_bus = len(opt_model_data.u_pu_max)
        n_line = len(opt_model_data.s_line)
        self.n_transformer = len(opt_model_data.s_transformer)
        n_gen = len(opt_model_data.p_min)

        # === Scaling factors ===
        p_abs_mean = np.mean(
            [
                max(abs(opt_model_data.p_max[i]), abs(opt_model_data.p_min[i]))
                for i in range(n_gen)
            ]
        )
        self.param_scale = 10 ** math.floor(
            math.log10(p_abs_mean)
        )  # Round down to nearest lower power of 10

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

        self.p_max.value = opt_model_data.p_max
        self.p_min.value = opt_model_data.p_min
        self.p_norm.value = opt_model_data.p_norm
        self.q_norm.value = opt_model_data.q_norm

        # Constraints
        cons = []

        # renewable gens
        for i in range(n_gen):
            cons += [self.p_gen[i] <= self.p_max[i] / self.param_scale]
            cons += [self.p_gen[i] >= self.p_min[i] / self.param_scale]
            has_q_limit = False
            if opt_model_data.s_inv[i] is not None:
                cons += [
                    cp.SOC(
                        1.0,
                        cp.hstack([self.p_gen[i], self.q_gen[i]])
                        / (opt_model_data.s_inv[i] / self.param_scale),
                    )
                ]
                has_q_limit = True
            if opt_model_data.pf_min[i] is not None:
                tan_phi = np.tan(np.arccos(opt_model_data.pf_min[i]))
                if opt_model_data.p_min[i] >= 0:  # generator
                    cons += [self.q_gen[i] <= tan_phi * self.p_gen[i]]
                    cons += [self.q_gen[i] >= -tan_phi * self.p_gen[i]]
                    has_q_limit = True
                elif opt_model_data.p_max[i] <= 0:  # load
                    cons += [self.q_gen[i] <= -tan_phi * self.p_gen[i]]
                    cons += [self.q_gen[i] >= tan_phi * self.p_gen[i]]
                    has_q_limit = True
                else:
                    # device can both generate and consume — drop PF constraint
                    print(
                        f"Warning: Controllable device {i} can both generate and consume (p_min < 0 < p_max). "
                        "Power factor constraint is skipped because the feasible region would be nonconvex."
                    )
            if opt_model_data.q_min[i] is not None:
                cons += [self.q_gen[i] >= opt_model_data.q_min[i] / self.param_scale]
            if opt_model_data.q_max[i] is not None:
                cons += [self.q_gen[i] <= opt_model_data.q_max[i] / self.param_scale]
            if (opt_model_data.q_min[i] is not None) and (
                opt_model_data.q_max[i] is not None
            ):
                has_q_limit = True
            if not has_q_limit:
                print(f"Warning: Controllable device {i} has unbounded reactive power.")
        # voltage
        cons += [
            self.u_pu_meas
            + sensitivities["du_dp"] @ (self.p_gen * self.param_scale - self.p_gen_last)
            + sensitivities["du_dq"] @ (self.q_gen * self.param_scale - self.q_gen_last)
            <= opt_model_data.u_pu_max
        ]
        cons += [
            self.u_pu_meas
            + sensitivities["du_dp"] @ (self.p_gen * self.param_scale - self.p_gen_last)
            + sensitivities["du_dq"] @ (self.q_gen * self.param_scale - self.q_gen_last)
            >= opt_model_data.u_pu_min
        ]

        # line
        for l in range(n_line):
            s_line = opt_model_data.s_line[l]
            cons += [
                cp.SOC(
                    1.0,
                    cp.hstack(
                        [
                            self.P_line_meas[l]
                            + sensitivities["dP_line_dp"][l, :]
                            @ (self.p_gen * self.param_scale - self.p_gen_last)
                            + sensitivities["dP_line_dq"][l, :]
                            @ (self.q_gen * self.param_scale - self.q_gen_last),
                            self.Q_line_meas[l]
                            + sensitivities["dQ_line_dp"][l, :]
                            @ (self.p_gen * self.param_scale - self.p_gen_last)
                            + sensitivities["dQ_line_dq"][l, :]
                            @ (self.q_gen * self.param_scale - self.q_gen_last),
                        ]
                    )
                    / s_line,
                )
            ]

        # transformer
        if self.n_transformer >= 1:
            for t in range(self.n_transformer):
                s_transformer = opt_model_data.s_transformer[t]
                cons += [
                    cp.SOC(
                        1.0,
                        cp.hstack(
                            [
                                self.P_transformer_meas[t]
                                + sensitivities["dP_transformer_dp"][t, :]
                                @ (self.p_gen * self.param_scale - self.p_gen_last)
                                + sensitivities["dP_transformer_dq"][t, :]
                                @ (self.q_gen * self.param_scale - self.q_gen_last),
                                self.Q_transformer_meas[t]
                                + sensitivities["dQ_transformer_dp"][t, :]
                                @ (self.p_gen * self.param_scale - self.p_gen_last)
                                + sensitivities["dQ_transformer_dq"][t, :]
                                @ (self.q_gen * self.param_scale - self.q_gen_last),
                            ]
                        )
                        / s_transformer,
                    )
                ]

        # Objective
        grad_p = 2.0 * cp.multiply(opt_model_data.c2_p, (self.p_gen_last - self.p_norm))
        grad_p += opt_model_data.c1_p
        grad_q = 2.0 * cp.multiply(opt_model_data.c2_q, (self.q_gen_last - self.q_norm))
        grad_q += opt_model_data.c1_q

        obj = cp.Minimize(
            cp.sum_squares(
                self.p_gen - (self.p_gen_last - alpha * grad_p) / self.param_scale
            )
            + cp.sum_squares(
                self.q_gen - (self.q_gen_last - alpha * grad_q) / self.param_scale
            )
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
            self.prob.solve(solver=getattr(cp, self.solver), **self.solver_kwargs)
        except cp.error.SolverError as e:
            print(f"Solver error: {e}")
            print("Returning previous generator setpoints.")
            return np.column_stack((param_dict["p_gen_last"], param_dict["q_gen_last"]))
        except Exception as e:
            print(f"Unexpected error: {e}")
            return np.column_stack((param_dict["p_gen_last"], param_dict["q_gen_last"]))

        # Check solver status
        if self.prob.status == cp.OPTIMAL:
            return (
                np.column_stack((self.p_gen.value, self.q_gen.value)) * self.param_scale
            )
        if self.prob.status == cp.OPTIMAL_INACCURATE:
            print("Solver finished with status: OPTIMAL_INACCURATE")
            return (
                np.column_stack((self.p_gen.value, self.q_gen.value)) * self.param_scale
            )
        else:
            print(
                f"Solver finished with status: {self.prob.status}. Returning previous generator setpoints."
            )
            return np.column_stack((param_dict["p_gen_last"], param_dict["q_gen_last"]))
