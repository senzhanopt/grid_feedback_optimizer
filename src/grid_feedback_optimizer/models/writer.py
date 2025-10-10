import json
from datetime import datetime
import numpy as np
from power_grid_model import ComponentType

def network_states_to_dict(output_data):
    """
    Convert the network output_data to a JSON-serializable dictionary.
    """
    net_dict = {
        'voltages': output_data[ComponentType.node]["u_pu"].tolist(),
        'line_loading': output_data[ComponentType.line]["loading"].tolist()
    }
    if ComponentType.transformer in output_data:
        net_dict['transformer_loading'] = output_data[ComponentType.transformer]["loading"].tolist()
    return net_dict

def setpoints_to_dict(gen_update: np.ndarray):
    """
    Convert generator updates to a dictionary.
    """
    gen_dict = {
        'p': gen_update[:,0].tolist(),
        'q': gen_update[:,1].tolist()
    }
    return gen_dict

def save_results(optimized_output_data, optimized_gen: np.ndarray, iterates=None, output_file="optimization_results.json"):
    """
    Save the results of the grid optimization in JSON format.
    
    Parameters:
    - output_data: final network state from solve()
    - gen_update: final generator setpoints from solve()
    - iterates: optional list of per-iteration states
    - output_file: path to save JSON
    """

    data = {
        "status": "success",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "iterations": len(iterates) if iterates else 0
        },
        "optimized_setpoints": setpoints_to_dict(optimized_gen),
        "network_states": network_states_to_dict(optimized_output_data),
        "iterates": []
    }

    # Include iteration history if provided
    if iterates:
        for it in iterates:
            data["iterates"].append({
                "iteration": it["iteration"],
                "setpoints": setpoints_to_dict(it["gen_update"]),
                "network_state": network_states_to_dict(it["output_data"])
            })

    # Write JSON file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Results saved to {output_file}")


