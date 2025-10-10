from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class SolveResults:
    final_output: Dict[Any, Any]
    final_gen_update: np.ndarray
    iterations: List[Dict[str, Any]]