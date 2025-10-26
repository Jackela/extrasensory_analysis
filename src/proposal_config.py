"""Proposal-aligned configuration for ExtraSensory analysis.

Decoupled configuration matching research proposal specifications:
- Tri-axis composite feature (SMA+variance 60/40 weight)
- Quintile discretization (5 bins)
- Per-subject z-score normalization
- CTE with 24-hour bins using STRATIFIED-TE with data-level lag
- TE/STE with JIDT v1.5 native 6-arg initialise for tau
- AIS-based k selection per subject
- Delta = A->S - S->A
- BH-FDR correction per (family, tau)
"""
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    composite_weights: Dict[str, float] = None  # {'sma': 0.6, 'variance': 0.4}
    n_bins: int = 5  # Quintile discretization
    normalization: str = 'per_subject_zscore'
    
    def __post_init__(self):
        if self.composite_weights is None:
            self.composite_weights = {'sma': 0.6, 'variance': 0.4}


@dataclass
class MethodConfig:
    """Information-theoretic method configuration."""
    # TE configuration
    te_use_native_tau: bool = True  # JIDT v1.5 6-arg initialise
    te_tau_values: List[int] = None
    
    # CTE configuration
    cte_hour_bins: int = 24
    cte_use_stratified: bool = True  # STRATIFIED-TE with data-level lag
    cte_tau_values: List[int] = None
    
    # STE configuration
    ste_use_native_tau: bool = True
    ste_ordinal_dim: int = 3
    ste_ordinal_delay: int = 1
    ste_tau_values: List[int] = None
    
    # Shared configuration
    k_selection_method: str = 'AIS'  # Use AIS to select k per subject
    k_range: List[int] = None  # k ∈ {1..6}
    num_surrogates: int = 1000
    delta_formula: str = 'A2S_minus_S2A'  # Delta = A->S - S->A
    
    def __post_init__(self):
        if self.te_tau_values is None:
            self.te_tau_values = [1, 2]
        if self.cte_tau_values is None:
            self.cte_tau_values = [1, 2]
        if self.ste_tau_values is None:
            self.ste_tau_values = [1, 2]
        if self.k_range is None:
            self.k_range = list(range(1, 7))  # {1, 2, 3, 4, 5, 6}


@dataclass
class StatisticsConfig:
    """Statistical testing configuration."""
    fdr_method: str = 'BH'  # Benjamini-Hochberg
    fdr_grouping: str = 'family_x_tau'  # Per (family, tau)
    alpha: float = 0.05


@dataclass
class ComputeConfig:
    """Computational resource configuration."""
    jvm_heap_min: str = '32g'
    jvm_heap_max: str = '48g'
    jvm_gc: str = 'G1GC'
    jvm_gc_pause_ms: int = 200
    concurrency: int = 1  # Single-threaded
    checkpoint_frequency: str = 'per_user_method'


@dataclass
class OutputConfig:
    """Output and artifact configuration."""
    out_dir: Path = None
    create_timestamp: bool = True
    save_formats: List[str] = None  # ['csv', 'parquet']
    generate_figures: bool = True
    heartbeat_file: str = 'status.json'
    
    def __post_init__(self):
        if self.save_formats is None:
            self.save_formats = ['csv', 'parquet']
        if self.out_dir is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M')
            self.out_dir = Path(f'analysis/out/proposal_align_{ts}')


@dataclass
class ProposalConfig:
    """Complete proposal-aligned configuration."""
    feature: FeatureConfig = None
    methods: MethodConfig = None
    statistics: StatisticsConfig = None
    compute: ComputeConfig = None
    output: OutputConfig = None
    
    # Data configuration
    feature_modes: List[str] = None
    n_users: int = 60  # Can override for smoke tests
    
    def __post_init__(self):
        if self.feature is None:
            self.feature = FeatureConfig()
        if self.methods is None:
            self.methods = MethodConfig()
        if self.statistics is None:
            self.statistics = StatisticsConfig()
        if self.compute is None:
            self.compute = ComputeConfig()
        if self.output is None:
            self.output = OutputConfig()
        if self.feature_modes is None:
            self.feature_modes = ['composite', 'sma_only', 'variance_only', 'magnitude_only']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'feature': {
                'composite_weights': self.feature.composite_weights,
                'n_bins': self.feature.n_bins,
                'normalization': self.feature.normalization
            },
            'methods': {
                'te_tau_values': self.methods.te_tau_values,
                'cte_hour_bins': self.methods.cte_hour_bins,
                'cte_use_stratified': self.methods.cte_use_stratified,
                'k_selection_method': self.methods.k_selection_method,
                'k_range': self.methods.k_range,
                'num_surrogates': self.methods.num_surrogates,
                'delta_formula': self.methods.delta_formula
            },
            'statistics': {
                'fdr_method': self.statistics.fdr_method,
                'fdr_grouping': self.statistics.fdr_grouping,
                'alpha': self.statistics.alpha
            },
            'compute': {
                'heap': f'-Xms{self.compute.jvm_heap_min} -Xmx{self.compute.jvm_heap_max}',
                'gc': self.compute.jvm_gc,
                'concurrency': self.compute.concurrency
            },
            'output': {
                'out_dir': str(self.output.out_dir),
                'save_formats': self.output.save_formats
            },
            'data': {
                'feature_modes': self.feature_modes,
                'n_users': self.n_users
            }
        }


def create_smoke_config() -> ProposalConfig:
    """Create configuration for smoke test (2 users, composite only)."""
    config = ProposalConfig()
    config.n_users = 2
    config.feature_modes = ['composite']
    config.output.out_dir = Path('analysis/out/smoke_test_' + datetime.now().strftime('%Y%m%d_%H%M'))
    return config


def create_full_config() -> ProposalConfig:
    """Create configuration for full 60-user analysis."""
    return ProposalConfig()
