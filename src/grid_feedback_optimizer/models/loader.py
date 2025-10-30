import json
from pathlib import Path

import pandas as pd

from grid_feedback_optimizer.models.network import (
    Bus,
    Line,
    Load,
    Network,
    RenewGen,
    Source,
    Transformer,
)


def load_network(file_path: str | Path) -> Network:
    """
    Load a network from a JSON file and validate it using Pydantic.

    Args:
        file_path: Path to the JSON network file.

    Returns:
        Network: Validated Network object.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Network file not found: {file_path}")

    with file_path.open("r") as f:
        raw_data = json.load(f)

    # Pydantic will validate all fields
    network = Network(**raw_data)
    return network


def load_network_from_excel(file_path: str | Path) -> Network:
    """
    Load a Network object from an Excel file where each sheet corresponds
    to a component type (e.g., 'buses', 'lines', etc.).

    Missing sheets are treated as empty lists.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    # Read all sheets into a dict of DataFrames
    sheets = pd.read_excel(file_path, sheet_name=None)

    def load_sheet(name: str, model):
        if name not in sheets:
            return []
        df = sheets[name]
        return [model(**rec) for rec in df.to_dict(orient="records")]

    network = Network(
        buses=load_sheet("buses", Bus),
        lines=load_sheet("lines", Line),
        transformers=load_sheet("transformers", Transformer),
        sources=load_sheet("sources", Source),
        renew_gens=load_sheet("renew_gens", RenewGen),
        loads=load_sheet("loads", Load),
    )

    return network
