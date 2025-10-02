from grid_feedback_optimizer.models.network import Network
import numpy as np
import pandas as pd
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
from power_grid_model.validation import assert_valid_input_data


class PowerFlowSolver:
    """
    Run a power flow simulation and return bus voltages and line currents.
    """
    def __init__(self, network: Network):
        self.build_network(network)

    def build_network(self, network):
        id_count = 0
        # node
        n_bus = len(network.buses)
        node = initialize_array(DatasetType.input, ComponentType.node, n_bus)
        node["id"] = np.arange(id_count, id_count + n_bus)
        node["u_rated"] = [bus.u_rated for bus in network.buses]
        id_count += n_bus

        # line
        n_line = len(network.lines)
        line = initialize_array(DatasetType.input, ComponentType.line, n_line)
        line["id"] = np.arange(id_count, id_count + n_line)
        line["from_node"] = [line.from_bus for line in network.lines]
        line["from_status"] = [1] * n_line
        line["to_status"] = [1] * n_line
        line["to_node"] = [line.to_bus for line in network.lines]
        line["r1"] = [line.r1 for line in network.lines]
        line["x1"] = [line.x1 for line in network.lines]
        line["c1"] = [line.c1 for line in network.lines]
        line["tan1"] = [line.tan1 for line in network.lines]
        line["i_n"] = [line.i_n for line in network.lines]
        id_count += n_line

        # source
        n_source = len(network.sources)
        source = initialize_array(DatasetType.input, ComponentType.source, n_source)
        source["id"] = np.arange(id_count, id_count + n_source)
        source["node"] = [source.bus for source in network.sources]
        source["status"] = [1] * n_source
        source["u_ref"] = [source.u_ref_pu for source in network.sources]
        id_count += n_source

        # load
        n_load = len(network.loads)
        sym_load = initialize_array(DatasetType.input, ComponentType.sym_load, n_load)
        sym_load["id"] = np.arange(id_count, id_count + n_load)
        sym_load["node"] = [load.bus for load in network.loads]
        sym_load["status"] = [1] * n_load
        sym_load["type"] = [LoadGenType.const_power] * n_load
        sym_load["p_specified"] = [load.p_norm for load in network.loads]
        sym_load["q_specified"] = [load.q_norm for load in network.loads]
        id_count += n_load

        # generator
        n_gen = len(network.renew_gens)
        sym_gen = initialize_array(DatasetType.input, ComponentType.sym_gen, n_gen)
        sym_gen["id"] = np.arange(id_count, id_count + n_gen)
        sym_gen["node"] = [gen.bus for gen in network.renew_gens]
        sym_gen["status"] = [1] * n_gen
        sym_gen["type"] = [LoadGenType.const_power] * n_gen
        sym_gen["p_specified"] = [gen.p_norm for gen in network.renew_gens]
        sym_gen["q_specified"] = [gen.q_norm for gen in network.renew_gens]
        id_count += n_gen

        # all
        input_data = {
            ComponentType.node: node,
            ComponentType.line: line,
            ComponentType.source: source,
            ComponentType.sym_load: sym_load,
            ComponentType.sym_gen: sym_gen,
        }
        
        # validation
        assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow)

        model = PowerGridModel(input_data)
        output_data = model.calculate_power_flow(
            symmetric=True, error_tolerance=1e-8, max_iterations=20, calculation_method=CalculationMethod.newton_raphson
        )

        self.n_gen = n_gen
        self.n_load = n_load
        self.model = model

        return output_data
    
    def run(self, gen_update: np.ndarray = None):
        """
        Re-run power flow in optimization iterations.
        The size of the arrays should match the total numbers of gens.
        """
        update_data_no_id = {}
        if gen_update is not None:
            update_sym_gen_no_id = initialize_array(DatasetType.update, ComponentType.sym_gen, self.n_gen)
            update_sym_gen_no_id["p_specified"] = gen_update[:, 0]
            update_sym_gen_no_id["q_specified"] = gen_update[:, 1]
            update_data_no_id[ComponentType.sym_gen] = update_sym_gen_no_id

        # rerun
        self.model.update(update_data = update_data_no_id)
        output_data = self.model.calculate_power_flow(
            symmetric=True, error_tolerance=1e-8, max_iterations=20, calculation_method=CalculationMethod.newton_raphson
        )

        return output_data
    
    def obtain_sensitivity(self):

        pass