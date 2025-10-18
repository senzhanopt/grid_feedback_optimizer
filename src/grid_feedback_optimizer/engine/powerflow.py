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

    def build_network(self, network: Network):
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

        # transformer
        n_transformer = len(network.transformers)
        if n_transformer >= 1:
            transformer = initialize_array(DatasetType.input, ComponentType.transformer, n_transformer)
            transformer["id"] = np.arange(id_count, id_count + n_transformer)
            transformer["from_node"] = [transformer.from_bus for transformer in network.transformers]
            transformer["to_node"] = [transformer.to_bus for transformer in network.transformers]
            transformer["from_status"] = [1] * n_transformer
            transformer["to_status"] = [1] * n_transformer
            transformer["u1"] = [transformer.u1 for transformer in network.transformers]
            transformer["u2"] = [transformer.u2 for transformer in network.transformers]
            transformer["sn"] = [transformer.sn for transformer in network.transformers]
            transformer["uk"] = [transformer.uk for transformer in network.transformers]
            transformer["pk"] = [transformer.pk for transformer in network.transformers]
            transformer["i0"] = [transformer.i0 for transformer in network.transformers]
            transformer["p0"] = [transformer.p0 for transformer in network.transformers]
            transformer["winding_from"] = [transformer.winding_from for transformer in network.transformers]
            transformer["winding_to"] = [transformer.winding_to for transformer in network.transformers]
            transformer["clock"] = [transformer.clock for transformer in network.transformers]
            transformer["tap_side"] = [transformer.tap_side for transformer in network.transformers]
            transformer["tap_pos"] = [transformer.tap_pos for transformer in network.transformers]
            transformer["tap_min"] = [transformer.tap_min for transformer in network.transformers]
            transformer["tap_max"] = [transformer.tap_max for transformer in network.transformers]
            transformer["tap_size"] = [transformer.tap_size for transformer in network.transformers]
            id_count += n_transformer


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
        if n_transformer >= 1:
            input_data[ComponentType.transformer] = transformer
        
        # validation
        assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow)

        model = PowerGridModel(input_data)
        output_data = model.calculate_power_flow(
            symmetric=True, error_tolerance=1e-8, max_iterations=20, calculation_method=CalculationMethod.newton_raphson
        )

        self.n_bus = n_bus
        self.n_line = n_line
        self.n_transformer = n_transformer
        self.n_gen = n_gen
        self.n_load = n_load
        self.model = model
        self.base_output_data = output_data
        self.base_p_gen = sym_gen["p_specified"]
        self.base_q_gen = sym_gen["q_specified"]
        self.u_pu_max = np.array([bus.u_pu_max for bus in network.buses])
        self.u_pu_min = np.array([bus.u_pu_min for bus in network.buses])
    
    @property
    def is_congested(self):
        """
        Check if the network is congested.
        Returns True if any of the following occur:
        - Bus voltage exceeds limits (u_pu_max or u_pu_min)
        - Line or transformer loading exceeds 1.0 (100%)
        """
        # Bus voltage check
        bus_voltages = self.base_output_data[ComponentType.node]["u_pu"]
        voltage_violation = np.any(bus_voltages > self.u_pu_max) or np.any(bus_voltages < self.u_pu_min)
        
        # Line loading check
        line_loading = self.base_output_data[ComponentType.line]["loading"]
        line_overload = np.any(line_loading > 1.0)

        # Transformer loading check
        if self.n_transformer >= 1:
            transformer_loading = self.base_output_data[ComponentType.transformer]["loading"]
            transformer_overload = np.any(transformer_loading > 1.0)
        else:
            transformer_overload = False

        # Return True if any violation occurs
        return voltage_violation or line_overload or transformer_overload
        

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
    
    def obtain_sensitivity(self, delta_p: float = 1.0, delta_q: float = 1.0, loading_meas_side: str = "from"):
        """
        Compute sensitivities of bus voltages and line/transformer power flows to small perturbations
        in generator power injections (p and q) around the default operating point.

        Parameters
        ----------
        delta_p : float
            Active power perturbation (W).
        delta_q : float
            Reactive power perturbation (VAr).
        loading_meas_side: str
            From which side branches are monitored: "from" or "to".

        Returns
        -------
        sensitivities : dict
            {
            "du_dp": array (n_bus, n_gen),
            "du_dq": array (n_bus, n_gen),
            "dP_line_dp": array (n_line, n_gen),
            "dQ_line_dp": array (n_line, n_gen),
            "dP_line_dq": array (n_line, n_gen),
            "dQ_line_dq": array (n_line, n_gen),
            "dP_transformer_dp": array (n_transformer, n_gen),
            "dQ_transformer_dp": array (n_transformer, n_gen),
            "dP_transformer_dq": array (n_transformer, n_gen),
            "dQ_transformer_dq": array (n_transformer, n_gen)
            }
        """
        # Base operating point
        u_pu_base = np.array(self.base_output_data[ComponentType.node]["u_pu"])
        P_line_base = np.array(self.base_output_data[ComponentType.line]["p_"+loading_meas_side])
        Q_line_base = np.array(self.base_output_data[ComponentType.line]["q_"+loading_meas_side])
        if self.n_transformer >= 1:
            P_transformer_base = np.array(self.base_output_data[ComponentType.transformer]["p_"+loading_meas_side])
            Q_transformer_base = np.array(self.base_output_data[ComponentType.transformer]["q_"+loading_meas_side])
        gen_base = np.column_stack((self.base_p_gen, self.base_q_gen))

        # Allocate arrays
        du_dp = np.zeros((self.n_bus, self.n_gen))
        du_dq = np.zeros((self.n_bus, self.n_gen))
        dP_line_dp = np.zeros((self.n_line, self.n_gen))
        dQ_line_dp = np.zeros((self.n_line, self.n_gen))
        dP_line_dq = np.zeros((self.n_line, self.n_gen))
        dQ_line_dq = np.zeros((self.n_line, self.n_gen))
        if self.n_transformer >= 1:
            dP_transformer_dp = np.zeros((self.n_transformer, self.n_gen))
            dQ_transformer_dp = np.zeros((self.n_transformer, self.n_gen))
            dP_transformer_dq = np.zeros((self.n_transformer, self.n_gen))
            dQ_transformer_dq = np.zeros((self.n_transformer, self.n_gen))

        # Loop over generators
        for g in range(self.n_gen):
            # perturb p
            gen_base[g, 0] += delta_p
            output_p = self.run(gen_update=gen_base)
            u_pu_p = np.array(output_p[ComponentType.node]["u_pu"])
            P_line_p = np.array(output_p[ComponentType.line]["p_"+loading_meas_side])
            Q_line_p = np.array(output_p[ComponentType.line]["q_"+loading_meas_side])
            if self.n_transformer >= 1:
                P_transformer_p = np.array(output_p[ComponentType.transformer]["p_"+loading_meas_side])
                Q_transformer_p = np.array(output_p[ComponentType.transformer]["q_"+loading_meas_side])

            du_dp[:, g] = (u_pu_p - u_pu_base) / delta_p
            dP_line_dp[:, g] = (P_line_p - P_line_base) / delta_p
            dQ_line_dp[:, g] = (Q_line_p - Q_line_base) / delta_p
            if self.n_transformer >= 1:
                dP_transformer_dp[:, g] = (P_transformer_p - P_transformer_base) / delta_p
                dQ_transformer_dp[:, g] = (Q_transformer_p - Q_transformer_base) / delta_p
            
            # recover to the base power
            gen_base[g, 0] -= delta_p

            # perturb q
            gen_base[g, 1] += delta_q
            output_q = self.run(gen_update=gen_base)
            u_pu_q = np.array(output_q[ComponentType.node]["u_pu"])
            P_line_q = np.array(output_q[ComponentType.line]["p_"+loading_meas_side])
            Q_line_q = np.array(output_q[ComponentType.line]["q_"+loading_meas_side])
            if self.n_transformer >= 1:
                P_transformer_q = np.array(output_q[ComponentType.transformer]["p_"+loading_meas_side])
                Q_transformer_q = np.array(output_q[ComponentType.transformer]["q_"+loading_meas_side])

            du_dq[:, g] = (u_pu_q - u_pu_base) / delta_q
            dP_line_dq[:, g] = (P_line_q - P_line_base) / delta_q
            dQ_line_dq[:, g] = (Q_line_q - Q_line_base) / delta_q
            if self.n_transformer >= 1:
                dP_transformer_dq[:, g] = (P_transformer_q - P_transformer_base) / delta_q
                dQ_transformer_dq[:, g] = (Q_transformer_q - Q_transformer_base) / delta_q
            
            # recover to the base power
            gen_base[g, 1] -= delta_q

        sensitivities = {
            "du_dp": du_dp,
            "du_dq": du_dq,
            "dP_line_dp": dP_line_dp,
            "dQ_line_dp": dQ_line_dp,
            "dP_line_dq": dP_line_dq,
            "dQ_line_dq": dQ_line_dq
        }

        if self.n_transformer >= 1:
            sensitivities["dP_transformer_dp"] = dP_transformer_dp
            sensitivities["dQ_transformer_dp"] = dQ_transformer_dp
            sensitivities["dP_transformer_dq"] = dP_transformer_dq
            sensitivities["dQ_transformer_dq"] = dQ_transformer_dq

        return sensitivities
