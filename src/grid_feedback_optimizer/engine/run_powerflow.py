from grid_feedback_optimizer.models.network import Network
import numpy as np
from power_grid_model import (
    CalculationMethod,
    CalculationType,
    ComponentAttributeFilterOptions,
    ComponentType,
    DatasetType,
    LoadGenType,
    PowerGridModel,
    attribute_dtype,
    initialize_array,
)

def run_powerflow(network: Network) -> dict:
    """
    Run a power flow simulation and return bus voltages and line currents.
    """
    n_buses = len(network.buses)
    # node
    node = initialize_array(DatasetType.input, ComponentType.node, n_buses)
    node["id"] = np.arange(0, n_buses)
    node["u_rated"] = [bus.u_rated for bus in network.buses]
        

 