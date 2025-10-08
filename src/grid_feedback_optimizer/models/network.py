from typing import Optional, List
from pydantic import BaseModel, Field
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
    index: int
    bus: int
    p_max: float
    s_inv: float
    p_min: Optional[float] = 0.0
    q_norm: Optional[float] = 0.0


class Load(BaseModel):
    index: int
    bus: int
    p_norm: float
    q_norm: float


# -------- Top-Level Container -------- #
class Network(BaseModel):
    buses: List[Bus]
    lines: List[Line]
    transformers: List[Transformer]
    sources: List[Source]
    renew_gens: List[RenewGen]
    loads: List[Load]