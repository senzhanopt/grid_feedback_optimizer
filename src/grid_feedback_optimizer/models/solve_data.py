from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
import json
from power_grid_model import ComponentType


class SolveResults(BaseModel):
    """
    Container for optimization and power flow results.
    Provides structured access and convenient save/load utilities.
    """

    final_output: Dict[Any, Any] # same output as power-grid-model
    final_gen_update: np.ndarray
    iterations: List[Dict[str, Any]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _network_states_to_dict(output_data: Dict[Any, Any]) -> Dict[str, Any]:
        """Convert the network output_data to a JSON-serializable dictionary."""
        net_dict = {
            "voltages": output_data[ComponentType.node]["u_pu"].tolist(),
            "line_loading": output_data[ComponentType.line]["loading"].tolist()
        }
        if ComponentType.transformer in output_data:
            net_dict["transformer_loading"] = output_data[ComponentType.transformer]["loading"].tolist()
        return net_dict

    @staticmethod
    def _setpoints_to_dict(gen_update: np.ndarray) -> Dict[str, Any]:
        """Convert generator setpoints to a JSON-friendly format."""
        return {"p": gen_update[:, 0].tolist(), "q": gen_update[:, 1].tolist()}

    def save(self, output_file: str = "optimization_results.json") -> None:
        """Save the results of the optimization and power flow to a JSON file."""
        data = {
            "status": "success",
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "iterations": len(self.iterations),
            },
            "optimized_setpoints": self._setpoints_to_dict(self.final_gen_update),
            "network_states": self._network_states_to_dict(self.final_output),
            "iterates": [],
        }

        # Add per-iteration details (if recorded)
        for it in self.iterations:
            data["iterates"].append({
                "iteration": it["iteration"],
                "setpoints": self._setpoints_to_dict(it["gen_update"]),
                "network_states": self._network_states_to_dict(it["output_data"]),
            })

        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"✅ Results saved to {output_file}")

    def print_summary(self) -> None:
        """Print a concise summary of the optimization results."""
        print("\n📊 Optimization Summary:")
        print(f"Total iterations: {len(self.iterations)}")
        print(f"Final generator setpoints (p): {self.final_gen_update[:, 0]}")
        print(f"Final generator setpoints (q): {self.final_gen_update[:, 1]}")
        net_dict = self._network_states_to_dict(self.final_output)
        print(f"Node voltages (p.u.): {net_dict['voltages']}")
        print(f"Line loadings (max 1.0): {net_dict['line_loading']}")
        if "transformer_loading" in net_dict:
            print(f"Transformer loadings (max 1.0): {net_dict['transformer_loading']}")
        print("✅ Summary printed successfully.\n")