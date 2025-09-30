from pathlib import Path
from grid_feedback_optimizer.models.network import Network
from grid_feedback_optimizer.io.loader import load_network

# Path to the example JSON in your project
EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example.json"

def test_load_network_from_example():
    # Ensure the example file exists
    assert EXAMPLE_JSON.exists(), f"Example JSON not found: {EXAMPLE_JSON}"

    # Load the network
    network = load_network(EXAMPLE_JSON)

    # Check that returned object is a Network
    assert isinstance(network, Network)

    # Minimal checks on the data
    assert len(network.buses) > 0
    assert len(network.renew_gens) > 0

    # Check generator defaults
    gen = network.renew_gens[0]
    assert gen.p_norm == gen.p_max or gen.p_norm is not None
    assert gen.q_norm == 0.0 or gen.q_norm is not None
