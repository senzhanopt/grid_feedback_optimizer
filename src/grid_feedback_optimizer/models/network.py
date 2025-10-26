from typing import Optional, List
from pydantic import BaseModel, Field, model_validator
from power_grid_model import WindingType, BranchSide

# -------- Core Models -------- #
# unit in W, V, A

class Bus(BaseModel):
    index: int
    u_rated: float 
    u_pu_max: float
    u_pu_min: float


class Line(BaseModel):
    index: int
    from_bus: int
    to_bus: int
    r1: float
    x1: float
    c1: float
    tan1: float
    i_n: float

class Transformer(BaseModel):
    index: int
    from_bus: int
    to_bus: int
    u1: float
    u2: float
    sn: float
    uk: float
    pk: float
    i0: float
    p0: float
    winding_from: WindingType
    winding_to: WindingType
    clock: int
    tap_side: BranchSide
    tap_min: int
    tap_max: int
    tap_size: float
    tap_pos: Optional[int] = 0


class Source(BaseModel):
    index: int
    bus: int
    u_ref_pu: float


class RenewGen(BaseModel):
    """
    Represents a renewable generator or power-consuming device.
    
    Notes:
        - If p_max > 0 and p_min >= 0 → behaves as a generator.
        - If p_max <= 0 and p_min <= 0 → behaves as a load/consuming device.
        - Mixed cases (p_min < 0 < p_max) → flexible device (can consume or generate).
    """
    index: int
    bus: int
    p_max: float
    s_inv: Optional[float] = None
    q_min: Optional[float] = None
    q_max: Optional[float] = None
    ratio_q_to_p_min: Optional[float] = None
    ratio_q_to_p_max: Optional[float] = None
    p_min: Optional[float] = 0.0
    q_norm: Optional[float] = 0.0
    p_norm: Optional[float] = None  # computed after initialization

    @model_validator(mode="after")
    def compute_p_norm(self):
        """Compute p_norm only if not provided by user."""
        if self.p_norm is not None:
            return self
    
        if self.p_min >= 0:
            self.p_norm = self.p_max
        elif self.p_max <= 0:
            self.p_norm = self.p_min
        else:
            self.p_norm = 0.0
        return self
    
    c1_p: Optional[float] = 0.0
    c2_p: float = Field(1.0, ge=0, description="Quadratic cost coefficient (> 0)")
    c1_q: Optional[float] = 0.0
    c2_q: float = Field(0.1, ge=0, description="Quadratic cost coefficient (> 0)")


class Load(BaseModel):
    """
    Represents a non-controllable unit: either a load or a generator.
    """
    index: int
    bus: int
    p_norm: float
    q_norm: float


# -------- Top-Level Container -------- #
class Network(BaseModel):
    buses: List[Bus]
    lines: List[Line]
    transformers: List[Transformer] = []
    sources: List[Source]
    renew_gens: List[RenewGen]
    loads: List[Load]