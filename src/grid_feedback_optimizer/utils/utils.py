import pandas as pd
from power_grid_model import ComponentType

def print_component(output_data, component_type, title=None):
    """
    Pretty-print a component's output data as a DataFrame.
    """
    if title:
        print(f"------ {title} ------")
    print(pd.DataFrame(output_data[component_type]))
    print()
