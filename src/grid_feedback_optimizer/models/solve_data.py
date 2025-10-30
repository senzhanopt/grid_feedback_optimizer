import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from power_grid_model import ComponentType


class SolveResults(BaseModel):
    """
    Container for optimization and power flow results.
    Provides structured access and convenient save/load utilities.
    """

    final_output: Dict[Any, Any]  # same output as power-grid-model
    final_gen_update: np.ndarray
    iterations: List[Dict[str, Any]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _network_states_to_dict(output_data: Dict[Any, Any]) -> Dict[str, Any]:
        """Convert the network output_data to a JSON-serializable dictionary."""
        net_dict = {
            "voltages": output_data[ComponentType.node]["u_pu"].tolist(),
            "line_loading": output_data[ComponentType.line]["loading"].tolist(),
        }
        if ComponentType.transformer in output_data:
            net_dict["transformer_loading"] = output_data[ComponentType.transformer][
                "loading"
            ].tolist()
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
            data["iterates"].append(
                {
                    "iteration": it["iteration"],
                    "setpoints": self._setpoints_to_dict(it["gen_update"]),
                    "network_states": self._network_states_to_dict(it["output_data"]),
                }
            )

        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"✅ Results saved to {output_file}")

    def print_summary(self) -> None:
        """Print a concise summary of the optimization results."""
        print("\n📊 Optimization Summary:")
        print(f"Total iterations: {len(self.iterations)}")
        print(
            f"Final generator setpoints (p): {np.round(self.final_gen_update[:, 0],2)}"
        )
        print(
            f"Final generator setpoints (q): {np.round(self.final_gen_update[:, 1],2)}"
        )
        net_dict = self._network_states_to_dict(self.final_output)
        print(f"Node voltages (p.u.): {np.round(net_dict['voltages'],3)}")
        print(f"Line loadings (max 1.0): {np.round(net_dict['line_loading'],3)}")
        if "transformer_loading" in net_dict:
            print(
                f"Transformer loadings (max 1.0): {np.round(net_dict['transformer_loading'],3)}"
            )
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
        voltages = np.array(
            [it["output_data"][ComponentType.node]["u_pu"] for it in self.iterations]
        )  # shape: (n_iter, n_nodes)

        lines = np.array(
            [it["output_data"][ComponentType.line]["loading"] for it in self.iterations]
        )  # shape: (n_iter, n_lines)

        transformers = (
            np.array(
                [
                    it["output_data"][ComponentType.transformer]["loading"]
                    for it in self.iterations
                ]
            )
            if ComponentType.transformer in self.iterations[0]["output_data"]
            else None
        )

        gen_p = np.array([it["gen_update"][:, 0] for it in self.iterations])
        gen_q = np.array([it["gen_update"][:, 1] for it in self.iterations])

        # === Plot ===
        fig, axes = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
        fig.suptitle("Optimization Iteration Evolution", fontsize=16, weight="bold")

        # 1️⃣ Voltages
        axes[0].plot(iter_indices, voltages, marker="o")
        axes[0].set_ylabel("Voltage [p.u.]")
        axes[0].set_title("Bus Voltages")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # 2️⃣ Line Loadings
        axes[1].plot(iter_indices, lines, marker="o")
        axes[1].set_ylabel("Loading")
        axes[1].set_title("Line Loadings")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        # 3️⃣ Transformer Loadings (if available)
        if transformers is not None:
            axes[2].plot(iter_indices, transformers, marker="o")
            axes[2].set_ylabel("Loading")
            axes[2].set_title("Transformer Loadings")
        else:
            axes[2].text(
                0.5,
                0.5,
                "No transformers in model",
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )
        axes[2].grid(True, linestyle="--", alpha=0.5)

        # 4️⃣ Generator Active Power (P)
        axes[3].plot(iter_indices, gen_p, marker="o")
        axes[3].set_ylabel("p [W]")
        axes[3].set_title("Generator Active Power")
        axes[3].grid(True, linestyle="--", alpha=0.5)

        # 5️⃣ Generator Reactive Power (Q)
        axes[4].plot(iter_indices, gen_q, marker="o")
        axes[4].set_ylabel("q [VAR]")
        axes[4].set_xlabel("Iteration")
        axes[4].set_title("Generator Reactive Power")
        axes[4].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        plt.close(fig)


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

    @model_validator(mode="after")
    def check_lengths(self):
        """Ensure measurement and generator arrays have consistent lengths."""

        # --- 1️. Check generator data lengths ---
        if len(self.p_gen_last) != len(self.q_gen_last):
            raise ValueError(
                f"p_gen_last and q_gen_last must have the same length: "
                f"(got {len(self.p_gen_last)} vs {len(self.q_gen_last)})."
            )

        # --- 2️. Check line flow data lengths ---
        if len(self.P_line_meas) != len(self.Q_line_meas):
            raise ValueError(
                f"P_line_meas and Q_line_meas must have the same length: "
                f"(got {len(self.P_line_meas)} vs {len(self.Q_line_meas)})."
            )

        # --- 3️. Check transformer data lengths (if provided) ---
        if self.P_transformer_meas is not None and self.Q_transformer_meas is not None:
            if len(self.P_transformer_meas) != len(self.Q_transformer_meas):
                raise ValueError(
                    f"P_transformer_meas and Q_transformer_meas must have the same length: "
                    f"(got {len(self.P_transformer_meas)} vs {len(self.Q_transformer_meas)})."
                )

        return self

    def to_dict(self):
        """Convert to a plain dict (for compatibility with existing code)."""
        data = self.model_dump()
        # Remove None entries (if transformers don’t exist)
        return {k: v for k, v in data.items() if v is not None}


class OptimizationModelData(BaseModel):
    """Minimal structure needed for the optimizer."""

    # generator data
    p_min: np.ndarray
    p_max: np.ndarray
    q_min: Optional[np.ndarray] = None
    q_max: Optional[np.ndarray] = None
    s_inv: Optional[np.ndarray] = None
    pf_min: Optional[np.ndarray] = None

    c1_p: np.ndarray
    c2_p: np.ndarray
    c1_q: np.ndarray
    c2_q: np.ndarray

    p_norm: np.ndarray
    q_norm: np.ndarray

    # voltage and branch data
    u_pu_max: np.ndarray
    u_pu_min: np.ndarray
    s_line: np.ndarray
    s_transformer: Optional[np.ndarray] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def check_fill_optional_arrays(self):
        """Ensure optional arrays are initialized as arrays of None if missing."""
        # === Step 1: Generator array consistency ===
        n_gen = len(self.p_min)
        same_length_fields = [
            "p_max",
            "c1_p",
            "c2_p",
            "c1_q",
            "c2_q",
            "p_norm",
            "q_norm",
        ]
        for f in same_length_fields:
            arr = getattr(self, f)
            if len(arr) != n_gen:
                raise ValueError(
                    f"Array '{f}' has length {len(arr)}, expected {n_gen}."
                )
        # === Step 2: Voltage array consistency ===
        if len(self.u_pu_max) != len(self.u_pu_min):
            raise ValueError(
                f"u_pu_max and u_pu_min must have the same length: "
                f"(got {len(self.u_pu_max)} vs {len(self.u_pu_min)})."
            )

        # === Step 3: Fill missing optional generator arrays ===
        def make_none_array(x):
            return x if x is not None else np.array([None] * n_gen, dtype=object)

        self.q_min = make_none_array(self.q_min)
        self.q_max = make_none_array(self.q_max)
        self.s_inv = make_none_array(self.s_inv)
        self.pf_min = make_none_array(self.pf_min)

        if self.s_transformer is None:
            self.s_transformer = np.array([], dtype=float)

        return self
