from pathlib import Path

from grid_feedback_optimizer.models.loader import load_network, load_network_from_excel
from grid_feedback_optimizer.models.network import Network


def test_load_network_from_example():

    # Path to the example JSON in your project
    EXAMPLE_JSON = Path(__file__).parent.parent / "examples" / "simple_example.json"

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
    assert gen.q_norm == 0.0 or gen.q_norm is not None


def test_load_network_from_example_with_transformer():

    # Path to the example JSON in your project
    EXAMPLE_JSON = (
        Path(__file__).parent.parent
        / "examples"
        / "simple_example_with_transformer.json"
    )

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
    assert gen.q_norm == 0.0 or gen.q_norm is not None


def test_load_network_from_example_with_transformer_from_excel():

    # Path to the example JSON in your project
    EXAMPLE_EXCEL = (
        Path(__file__).parent.parent
        / "examples"
        / "simple_example_with_transformer.xlsx"
    )

    # Ensure the example file exists
    assert EXAMPLE_EXCEL.exists(), f"Example EXCEL not found: {EXAMPLE_EXCEL}"

    # Load the network
    network = load_network_from_excel(EXAMPLE_EXCEL)

    # Check that returned object is a Network
    assert isinstance(network, Network)

    # Minimal checks on the data
    assert len(network.buses) > 0
    assert len(network.renew_gens) > 0

    # Check generator defaults
    gen = network.renew_gens[0]
    assert gen.q_norm == 0.0 or gen.q_norm is not None
