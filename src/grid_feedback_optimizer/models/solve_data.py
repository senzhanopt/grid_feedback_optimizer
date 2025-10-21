from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
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
        print(f"Final generator setpoints (p): {np.round(self.final_gen_update[:, 0],2)}")
        print(f"Final generator setpoints (q): {np.round(self.final_gen_update[:, 1],2)}")
        net_dict = self._network_states_to_dict(self.final_output)
        print(f"Node voltages (p.u.): {np.round(net_dict['voltages'],3)}")
        print(f"Line loadings (max 1.0): {np.round(net_dict['line_loading'],3)}")
        if "transformer_loading" in net_dict:
            print(f"Transformer loadings (max 1.0): {np.round(net_dict['transformer_loading'],3)}")
        print("✅ Summary printed successfully.\n")

    def plot_iterations(self) -> None:
        """
        Plot evolution of voltages, line loadings, transformer loadings,
        and generator active/reactive powers over optimization iterations.

        Layout: 5 rows * 1 column (shared x-axis: iteration number)
        """
        if not self.iterations:
            print("⚠️ No iteration data available for plotting.")
            return

        iter_indices = [it["iteration"] for it in self.iterations]

        # === Collect data ===
        voltages = np.array([
            it["output_data"][ComponentType.node]["u_pu"] for it in self.iterations
        ])  # shape: (n_iter, n_nodes)

        lines = np.array([
            it["output_data"][ComponentType.line]["loading"] for it in self.iterations
        ])  # shape: (n_iter, n_lines)

        transformers = (
            np.array([it["output_data"][ComponentType.transformer]["loading"]
                    for it in self.iterations])
            if ComponentType.transformer in self.iterations[0]["output_data"]
            else None
        )

        gen_p = np.array([it["gen_update"][:, 0] for it in self.iterations])
        gen_q = np.array([it["gen_update"][:, 1] for it in self.iterations])

        # === Plot ===
        fig, axes = plt.subplots(5, 1, figsize=(10, 16), sharex=True)
        fig.suptitle("Optimization Iteration Evolution", fontsize=16, weight='bold')

        # 1️⃣ Voltages
        axes[0].plot(iter_indices, voltages, marker='o')
        axes[0].set_ylabel("Voltage [p.u.]")
        axes[0].set_title("Bus Voltages")
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # 2️⃣ Line Loadings
        axes[1].plot(iter_indices, lines, marker='o')
        axes[1].set_ylabel("Loading")
        axes[1].set_title("Line Loadings")
        axes[1].grid(True, linestyle='--', alpha=0.5)

        # 3️⃣ Transformer Loadings (if available)
        if transformers is not None:
            axes[2].plot(iter_indices, transformers, marker='o')
            axes[2].set_ylabel("Loading")
            axes[2].set_title("Transformer Loadings")
        else:
            axes[2].text(0.5, 0.5, "No transformers in model",
                        ha='center', va='center', fontsize=10, color='gray')
        axes[2].grid(True, linestyle='--', alpha=0.5)

        # 4️⃣ Generator Active Power (P)
        axes[3].plot(iter_indices, gen_p, marker='o')
        axes[3].set_ylabel("p [W]")
        axes[3].set_title("Generator Active Power")
        axes[3].grid(True, linestyle='--', alpha=0.5)

        # 5️⃣ Generator Reactive Power (Q)
        axes[4].plot(iter_indices, gen_q, marker='o')
        axes[4].set_ylabel("q [VAR]")
        axes[4].set_xlabel("Iteration")
        axes[4].set_title("Generator Reactive Power")
        axes[4].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

class OptimizationInputs(BaseModel):
    """
    Structured container for optimization inputs at each iteration.
    """
    u_pu_meas: np.ndarray
    P_line_meas: np.ndarray
    Q_line_meas: np.ndarray
    p_gen_last: np.ndarray
    q_gen_last: np.ndarray
    P_transformer_meas: Optional[np.ndarray] = None
    Q_transformer_meas: Optional[np.ndarray] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self):
        """Convert to a plain dict (for compatibility with existing code)."""
        data = self.model_dump()
        # Remove None entries (if transformers don’t exist)
        return {k: v for k, v in data.items() if v is not None}