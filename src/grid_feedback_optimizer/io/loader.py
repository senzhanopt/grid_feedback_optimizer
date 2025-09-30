import json
from pathlib import Path
from grid_feedback_optimizer.models.network import Network

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