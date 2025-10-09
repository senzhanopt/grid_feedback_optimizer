import pandas as pd
from power_grid_model import ComponentType

def print_component(output_data, component_type):
    """
    Pretty-print a component's output data as a DataFrame.
    """
    print(f"------ {component_type} result ------")
    if component_type == "node":
        print(pd.DataFrame(output_data[ComponentType.node]))
        print()
    elif component_type == "line":
        print(pd.DataFrame(output_data[ComponentType.line]))
        print()
    elif component_type == "transformer":
        print(pd.DataFrame(output_data[ComponentType.transformer]))
        print()       
