from pathlib import Path
from grid_feedback_optimizer.io.loader import load_network

# Path to your example JSON
json_file = Path(__file__).parent.parent / "examples" / "simple_example.json"

# Load network
network = load_network(json_file)

print(network.buses)
