import cvxpy as cp
import numpy as np

class RenewGenProjection:
    """
    Project points onto the feasible inverter operating region.
    Supports analytical (with p_min = 0) and CVXPY-based (free p_min) projections.
    """   
    def __init__(self, solver: str = "CLARABEL", **solver_kwargs):
        """Initialize the CVXPY problem for repeated projections."""
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self._setup_qp()

    # ---------------- CVXPY Projection Setup ----------------
    def _setup_qp(self):
        """Set up the CVXPY problem with Parameters for repeated solves."""
        self.p_var = cp.Variable()
        self.q_var = cp.Variable()
        self.p_max = cp.Parameter()
        self.p_min = cp.Parameter()
        self.s_inv = cp.Parameter()
        self.p = cp.Parameter()
        self.q = cp.Parameter()

        objective = cp.Minimize(cp.square(self.p_var - self.p) + cp.square(self.q_var - self.q))
        cons = [
            self.p_var >= self.p_min,
            self.p_var <= self.p_max,
            cp.SOC(self.s_inv, cp.hstack([self.p_var, self.q_var]))
        ]

        self.cvxpy_problem = cp.Problem(objective, cons)

    def opt_projection(self, p_max: float, p_min: float, s_inv: float, p: float, q: float):
        """Solve the CVXPY projection problem with updated parameter values."""
        self.p_max.value = p_max
        self.p_min.value = p_min
        self.s_inv.value = s_inv
        self.p.value = p
        self.q.value = q

        try:
            # Solve the problem
            self.cvxpy_problem.solve(solver=getattr(cp, self.solver), **self.solver_kwargs)
        except cp.error.SolverError as e:
            # Catch CVXPY solver-specific errors
            raise ValueError(f"CVXPY solver error: {e}, status={self.cvxpy_problem.status}")
        except Exception as e:
            # Catch any other unexpected errors
            raise ValueError(f"CVXPY projection failed due to unexpected error: {e}, status={self.cvxpy_problem.status}")

        # Check solver status
        if self.cvxpy_problem.status == cp.OPTIMAL:
            return np.array([self.p_var.value, self.q_var.value])
        else:
            raise ValueError(f"CVXPY projection did not converge to optimal solution, status={self.cvxpy_problem.status}")

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
        if (p**2 + q**2 >= s_inv**2) and (p >= 0) and (abs(q) >= q_max/p_max * p):
            scale = s_inv / np.sqrt(p**2 + q**2)
            return np.array([p, q]) * scale

        # Region B
        if (q >= q_max) and (q <= q_max/p_max * p):
            return np.array([p_max, q_max])
        if (q <= -q_max) and (q >= -q_max/p_max * p):
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
