"""Parameter dataclasses for JIDT analysis."""
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TEParams:
    """Transfer Entropy parameters."""
    base_source: int
    base_dest: int
    k_source: int
    k_dest: int
    tau: int = 1
    num_surrogates: int = 1000
    # Adaptive significance testing (optional)
    adaptive_stages: Optional[List[int]] = None
    early_stop_sig: Optional[float] = None
    early_stop_nonsig: Optional[float] = None
    seed: Optional[int] = None

@dataclass
class CTEParams:
    """Conditional Transfer Entropy parameters."""
    base_source: int
    base_dest: int
    base_cond: int
    k_source: int
    k_dest: int
    num_cond_bins: int
    tau: int = 1
    num_surrogates: int = 1000
    # Propagate adaptive settings to per-bin TE
    adaptive_stages: Optional[List[int]] = None
    early_stop_sig: Optional[float] = None
    early_stop_nonsig: Optional[float] = None
    seed: Optional[int] = None

@dataclass
class STEParams:
    """Symbolic Transfer Entropy parameters."""
    ordinal_dim: int = 3
    ordinal_delay: int = 1
    k_source: int = 2
    k_dest: int = 2
    tau: int = 1
    num_surrogates: int = 1000
    seed: Optional[int] = None
