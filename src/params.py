"""
Parameter dataclasses for JIDT analysis.

Defines configuration structures for TE, CTE, and Symbolic TE.
All fields are English-only and intended for serialization.

@module params
"""
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TEParams:
    """
    Transfer Entropy parameters.

    @property base_source {int} Alphabet size for source.
    @property base_dest {int} Alphabet size for destination.
    @property k_source {int} History for source.
    @property k_dest {int} History for destination.
    @property tau {int} Delay parameter.
    @property num_surrogates {int} Surrogates for significance.
    @property adaptive_stages {Optional[List[int]]} Staged surrogates (optional).
    @property early_stop_sig {Optional[float]} Early stop threshold for significance.
    @property early_stop_nonsig {Optional[float]} Early stop threshold for non-significance.
    @property seed {Optional[int]} RNG seed.
    """
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
    """
    Conditional Transfer Entropy parameters.

    @property base_source {int} Alphabet size for source.
    @property base_dest {int} Alphabet size for destination.
    @property base_cond {int} Alphabet size for conditioning variable.
    @property k_source {int} History for source.
    @property k_dest {int} History for destination.
    @property num_cond_bins {int} Number of conditional variables (historical, set to 1 for hour bin).
    @property tau {int} Delay parameter.
    @property num_surrogates {int} Surrogates for significance.
    @property adaptive_stages {Optional[List[int]]} Staged surrogates (optional).
    @property early_stop_sig {Optional[float]} Early stop threshold for significance.
    @property early_stop_nonsig {Optional[float]} Early stop threshold for non-significance.
    @property seed {Optional[int]} RNG seed.
    """
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
    """
    Symbolic Transfer Entropy parameters.

    @property ordinal_dim {int} Ordinal pattern dimension.
    @property ordinal_delay {int} Ordinal pattern delay.
    @property k_source {int} History for source (symbolic).
    @property k_dest {int} History for destination (symbolic).
    @property tau {int} Delay parameter.
    @property num_surrogates {int} Surrogates for significance.
    @property seed {Optional[int]} RNG seed.
    """
    ordinal_dim: int = 3
    ordinal_delay: int = 1
    k_source: int = 2
    k_dest: int = 2
    tau: int = 1
    num_surrogates: int = 1000
    seed: Optional[int] = None

@dataclass
class CTEKraskovParams:
    """
    Continuous (Kraskov) Conditional Transfer Entropy parameters.

    @property k_source {int} History for source.
    @property k_dest {int} History for destination.
    @property tau {int} Delay parameter.
    @property k_nn {int} Nearest neighbors for Kraskov estimator.
    @property num_surrogates {int} Surrogates for significance (if supported).
    @property seed {Optional[int]} RNG seed.
    """
    k_source: int
    k_dest: int
    tau: int = 1
    k_nn: int = 4
    num_surrogates: int = 1000
    seed: Optional[int] = None
