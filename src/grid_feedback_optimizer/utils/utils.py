import numpy as np
import pandas as pd
from power_grid_model import ComponentType

from grid_feedback_optimizer.models.network import Network
from grid_feedback_optimizer.models.solve_data import OptimizationModelData


class TransformerActivePowerTrackingCallback:
    def __init__(
        self, sensitivity: np.ndarray, alpha: float = 1.0, n_transformer: int = 1
    ):
        """
        Make sure the sensitivity is calculated using "from_side".
        """
        self.dual = np.zeros(n_transformer)
        self.n_transformer = n_transformer
        self.sensitivity = sensitivity
        self.alpha = alpha

    def __call__(
        self,
        grad_p: np.ndarray,
        grad_q: np.ndarray,
        P_transformer_meas_from: np.ndarray,
        target_power: np.ndarray | None = None,
    ):
        # update dual variable
        if target_power is None:
            self.dual = np.zeros(self.n_transformer)
        else:
            self.dual += (
                self.alpha * (P_transformer_meas_from - target_power) / 500.0
            )  # a simple scaling
            self.dual[self.dual < 0] = 0.0

        grad_p += self.sensitivity["dP_transformer_dp"].T @ self.dual
        grad_q += self.sensitivity["dP_transformer_dq"].T @ self.dual

        return grad_p, grad_q


def network_to_model_data(network: Network) -> OptimizationModelData:
    """Convert a Network object into OptimizationModelData."""
    return OptimizationModelData(
        p_min=np.array([g.p_min for g in network.renew_gens]),
        p_max=np.array([g.p_max for g in network.renew_gens]),
        q_min=np.array([g.q_min for g in network.renew_gens]),
        q_max=np.array([g.q_max for g in network.renew_gens]),
        s_inv=np.array([g.s_inv for g in network.renew_gens]),
        pf_min=np.array([g.pf_min for g in network.renew_gens]),
        c1_p=np.array([g.c1_p for g in network.renew_gens]),
        c2_p=np.array([g.c2_p for g in network.renew_gens]),
        c1_q=np.array([g.c1_q for g in network.renew_gens]),
        c2_q=np.array([g.c2_q for g in network.renew_gens]),
        p_norm=np.array([g.p_norm for g in network.renew_gens]),
        q_norm=np.array([g.q_norm for g in network.renew_gens]),
        u_pu_max=np.array([bus.u_pu_max for bus in network.buses]),
        u_pu_min=np.array([bus.u_pu_min for bus in network.buses]),
        s_line=np.array(
            [
                np.sqrt(3) * line.i_n * network.buses[line.from_bus].u_rated
                for line in network.lines
            ]
        ),
        s_transformer=(
            np.array([t.sn for t in network.transformers])
            if len(network.transformers) >= 1
            else None
        ),
    )
