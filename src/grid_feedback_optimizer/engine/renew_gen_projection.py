from functools import lru_cache

import cvxpy as cp
import numpy as np


class RenewGenProjection:
    """
    Project points onto the feasible inverter operating region.
    Supports analytical and CVXPY-based projections.
    """

    def __init__(self, solver: str = "CLARABEL", **solver_kwargs):
        """Initialize the CVXPY problem for repeated projections."""
        self.solver = solver
        self.solver_kwargs = solver_kwargs

    @staticmethod
    def analytic_projection(p_max: float, s_inv: float, p: float, q: float):
        """
        Project a point (p, q) onto the feasible operating region
        of a PV inverter defined by active power limit p_max and apparent power limit s_inv.

        Based on: Optimal Power Flow Pursuit (Appendix B) — setpoint update rule.

        Parameters
        ----------
        p_max : float
            Active power limit (p_max >= 0)
        s_inv : float
            Inverter apparent power capacity (s_inv >= 0)
        p, q : float
            Active and reactive power values to project

        Returns
        -------
        np.ndarray
            The projected point [p_proj, q_proj] lying within the feasible PV region.

        Raises
        ------
        ValueError
            If projection fails or inputs are inconsistent.
        """
        # Reactive power limit at maximum active power
        q_max = np.sqrt(s_inv**2 - p_max**2)

        # Handle degenerate case where active power limit is zero
        if p_max == 0.0:
            if q >= s_inv:
                return np.array([0.0, s_inv])
            elif q <= -s_inv:
                return np.array([0.0, -s_inv])
            else:
                return np.array([0.0, q])

        # Region Y
        if (p**2 + q**2 <= s_inv**2) and (0 <= p <= p_max):
            return np.array([p, q])

        # Region A
        if (p**2 + q**2 >= s_inv**2) and (p >= 0) and (abs(q) >= q_max / p_max * p):
            scale = s_inv / np.sqrt(p**2 + q**2)
            return np.array([p, q]) * scale

        # Region B
        if (q >= q_max) and (q <= q_max / p_max * p):
            return np.array([p_max, q_max])
        if (q <= -q_max) and (q >= -q_max / p_max * p):
            return np.array([p_max, -q_max])

        # Region C
        if (p >= p_max) and (-q_max <= q <= q_max):
            return np.array([p_max, q])

        # Region D
        if (p <= 0) and (-s_inv <= q <= s_inv):
            return np.array([0.0, q])

        # Region E
        if (p <= 0) and (q >= s_inv):
            return np.array([0.0, s_inv])
        if (p <= 0) and (q <= -s_inv):
            return np.array([0.0, -s_inv])

        # If none of the conditions match, projection failed
        raise ValueError(
            f"Projection failed for values: p_max={p_max:.3f}, s_inv={s_inv:.3f}, p={p:.3f}, q={q:.3f}"
        )

    # ---------------- CVXPY Projection Setup ----------------
    @staticmethod
    @lru_cache(maxsize=None)
    def _build_problem(
        has_s_inv: bool = True,
        has_pf_min: bool = False,
        has_q_min: bool = False,
        has_q_max: bool = False,
        is_generator: bool = True,
        is_load: bool = False,
    ):
        """Set up the CVXPY problem with Parameters for repeated solves."""
        p_var = cp.Variable()
        q_var = cp.Variable()
        p_max = cp.Parameter()
        p_min = cp.Parameter()
        p = cp.Parameter()
        q = cp.Parameter()

        vars = dict(p_var=p_var, q_var=q_var)
        params = dict(p=p, q=q, p_max=p_max, p_min=p_min)

        objective = cp.Minimize(cp.square(p_var - p) + cp.square(q_var - q))
        cons = [p_var >= p_min, p_var <= p_max]

        if has_s_inv:
            s_inv = cp.Parameter(nonneg=True)
            params["s_inv"] = s_inv
            cons += [cp.SOC(s_inv, cp.hstack([p_var, q_var]))]

        if has_pf_min:
            tan_angle = cp.Parameter(nonneg=True)
            params["tan_angle"] = tan_angle
            if is_generator:
                cons += [q_var <= p_var * tan_angle, q_var >= -p_var * tan_angle]
            elif is_load:
                cons += [q_var <= -p_var * tan_angle, q_var >= p_var * tan_angle]
            else:
                # device can both generate and consume — drop PF constraint
                print(
                    "Warning: Controllable device can both generate and consume (p_min < 0 < p_max). "
                    "Power factor constraint is skipped because the feasible region would be nonconvex."
                )
                if (
                    not has_s_inv
                ):  # the case when s_inv and pf_min are both None is handled as simple analytic projection
                    print("Warning: Controllable device has unbounded reactive power.")

        if has_q_min:
            q_min = cp.Parameter()
            params["q_min"] = q_min
            cons += [q_var >= q_min]
        if has_q_max:
            q_max = cp.Parameter()
            params["q_max"] = q_max
            cons += [q_var <= q_max]

        prob = cp.Problem(objective, cons)

        return prob, params, vars

    def projection(
        self,
        p_max: float,
        p_min: float,
        p: float,
        q: float,
        s_inv: float | None = None,
        pf_min: float | None = None,
        q_min: float | None = None,
        q_max: float | None = None,
    ):
        """Hybrid projection: analytic if simple, CVXPY otherwise."""

        has_s_inv = s_inv is not None
        has_pf_min = pf_min is not None
        has_q_min = q_min is not None
        has_q_max = q_max is not None
        is_generator = True if p_min >= 0 else False
        is_load = True if p_max <= 0 else False

        # Fast analytic case: rectangular p-q region
        if (not has_s_inv) and (not has_pf_min):
            q_min = q_min if has_q_min else -np.inf
            q_max = q_max if has_q_max else np.inf
            p_proj = np.clip(p, p_min, p_max)
            q_proj = np.clip(q, q_min, q_max)
            if (not has_q_min) or (not has_q_max):
                print("Warning: Controllable device has unbounded reactive power.")
            return np.array([p_proj, q_proj])

        # Fast analytic case: only s_inv
        if has_s_inv and (not has_pf_min) and (not has_q_min) and (not has_q_max):
            if p_min == 0 or (p_min < 0 and p_max > 0 and p >= 0):
                return self.analytic_projection(p_max=p_max, s_inv=s_inv, p=p, q=q)
            if p_max == 0 or (p_min < 0 and p_max > 0 and p < 0):
                return self.analytic_projection(
                    p_max=-p_min, s_inv=s_inv, p=-p, q=q
                ) * np.array([-1.0, 1.0])

        # all other cases where optimization problem is solved for projection
        prob, params, vars = self._build_problem(
            has_s_inv=has_s_inv,
            has_pf_min=has_pf_min,
            has_q_min=has_q_min,
            has_q_max=has_q_max,
            is_generator=is_generator,
            is_load=is_load,
        )

        # assign values
        params["p"].value = p
        params["q"].value = q
        params["p_max"].value = p_max
        params["p_min"].value = p_min
        if has_s_inv:
            params["s_inv"].value = s_inv
        if has_pf_min:
            params["tan_angle"].value = np.tan(np.arccos(pf_min))
        if has_q_max:
            params["q_max"].value = q_max
        if has_q_min:
            params["q_min"].value = q_min

        try:
            # Solve the problem
            prob.solve(solver=getattr(cp, self.solver), **self.solver_kwargs)
        except cp.error.SolverError as e:
            # Catch CVXPY solver-specific errors
            raise ValueError(f"CVXPY solver error: {e}, status={prob.status}")
        except Exception as e:
            # Catch any other unexpected errors
            raise ValueError(
                f"CVXPY projection failed due to unexpected error: {e}, status={prob.status}"
            )

        # Check solver status
        if prob.status == cp.OPTIMAL:
            return np.array([vars["p_var"].value, vars["q_var"].value])
        if prob.status == cp.OPTIMAL_INACCURATE:
            print("Solver finished with status: OPTIMAL_INACCURATE")
            return np.array([vars["p_var"].value, vars["q_var"].value])
        else:
            raise ValueError(
                f"CVXPY projection did not converge to optimal solution, status={prob.status}"
            )
