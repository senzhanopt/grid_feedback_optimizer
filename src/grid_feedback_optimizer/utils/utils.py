import pandas as pd
from power_grid_model import ComponentType
import numpy as np

class TransformerActivePowerTrackingCallback:
    def __init__(self, sensitivity: np.ndarray, alpha: float = 1.0, n_transformer: int = 1):
        """
        Make sure the sensitivity is calculated using "from_side".
        """
        self.dual = np.zeros(n_transformer)
        self.n_transformer = n_transformer
        self.sensitivity = sensitivity
        self.alpha = alpha

    def __call__(self, grad_p: np.ndarray, grad_q: np.ndarray,
                 P_transformer_meas_from: np.ndarray, target_power: np.ndarray | None = None):
        # update dual variable
        if target_power is None:
            self.dual = np.zeros(self.n_transformer)
        else:
            self.dual += self.alpha * (P_transformer_meas_from - target_power) / 500.0 # a simple scaling
            self.dual[self.dual < 0] = 0.0
        
        grad_p += self.sensitivity["dP_transformer_dp"].T @ self.dual
        grad_q += self.sensitivity["dP_transformer_dq"].T @ self.dual

        return grad_p, grad_q
        

        