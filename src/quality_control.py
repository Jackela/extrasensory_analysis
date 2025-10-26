# src/quality_control.py
# Data quality validation and threshold enforcement
# Centralizes all sample size requirements and quality checks

import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QualityAction(Enum):
    """Actions to take when quality thresholds are violated."""
    SKIP = "skip"  # Skip entire user/analysis
    WARN = "warn"  # Log warning, continue processing
    FILTER = "filter_bins"  # Filter problematic bins (CTE)
    ERROR = "error"  # Raise exception


class DataQualityError(Exception):
    """Raised when data quality is insufficient and action=skip."""
    pass


@dataclass
class MethodThresholds:
    """Quality thresholds for a specific analysis method."""
    min_samples: int
    min_samples_per_state: Optional[int] = None
    action: QualityAction = QualityAction.WARN
    rationale: str = ""


@dataclass
class CTEThresholds:
    """Special thresholds for Conditional Transfer Entropy."""
    min_total_samples: int
    min_bin_samples: int
    max_filtered_bins: int
    max_filtered_ratio: float
    action: QualityAction
    rationale: str = ""


@dataclass
class KSelectionThresholds:
    """Thresholds for k-selection quality control."""
    min_samples_per_state: int
    enforce: bool
    fallback_k: int
    max_k_absolute: int
    rationale: str = ""


class QualityController:
    """
    Validates data quality against configurable thresholds.
    
    Centralizes all sample size requirements and quality checks
    to ensure consistent enforcement across the analysis pipeline.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize quality controller from config dictionary.
        
        Args:
            config: Quality control configuration (from YAML)
        """
        qc = config.get('quality_control', {})
        
        # Global thresholds
        global_cfg = qc.get('global', {})
        self.min_total_samples = global_cfg.get('min_total_samples', 100)
        self.min_bin_samples = global_cfg.get('min_bin_samples', 30)
        self.global_action = QualityAction(global_cfg.get('action', 'warn'))
        
        # Method-specific thresholds
        self.te = self._parse_method_thresholds(qc.get('te', {}))
        self.ste = self._parse_method_thresholds(qc.get('ste', {}))
        self.gc = self._parse_method_thresholds(qc.get('gc', {}))
        
        # CTE special handling
        cte_cfg = qc.get('cte', {})
        self.cte = CTEThresholds(
            min_total_samples=cte_cfg.get('min_total_samples', 200),
            min_bin_samples=cte_cfg.get('min_bin_samples', 50),
            max_filtered_bins=cte_cfg.get('max_filtered_bins', 4),
            max_filtered_ratio=cte_cfg.get('max_filtered_ratio', 0.5),
            action=QualityAction(cte_cfg.get('action', 'filter_bins')),
            rationale=cte_cfg.get('rationale', '')
        )
        
        # k-selection thresholds
        k_cfg = qc.get('k_selection', {})
        self.k_selection = KSelectionThresholds(
            min_samples_per_state=k_cfg.get('min_samples_per_state', 25),
            enforce=k_cfg.get('enforce', True),
            fallback_k=k_cfg.get('fallback_k', 3),
            max_k_absolute=k_cfg.get('max_k_absolute', 6),
            rationale=k_cfg.get('rationale', '')
        )
        
        # Statistical settings
        stats = config.get('statistical', {})
        self.significance_level = stats.get('significance_level', 0.05)
        self.min_power = stats.get('min_power', 0.70)
        
        # Reporting settings
        report = config.get('reporting', {})
        self.log_warnings = report.get('log_quality_warnings', True)
        self.generate_report = report.get('generate_quality_report', False)
    
    def _parse_method_thresholds(self, cfg: Dict) -> MethodThresholds:
        """Parse method-specific threshold configuration."""
        return MethodThresholds(
            min_samples=cfg.get('min_samples', 100),
            min_samples_per_state=cfg.get('min_samples_per_state'),
            action=QualityAction(cfg.get('action', 'warn')),
            rationale=cfg.get('rationale', '')
        )
    
    def validate_global(self, n_samples: int, user_id: str) -> bool:
        """
        Validate global minimum sample requirements.
        
        Args:
            n_samples: Total sample count
            user_id: User identifier for logging
            
        Returns:
            True if validation passed
            
        Raises:
            DataQualityError: If action=skip and threshold violated
        """
        if n_samples < self.min_total_samples:
            msg = f"Insufficient samples: {n_samples} < {self.min_total_samples}"
            
            if self.global_action == QualityAction.SKIP:
                raise DataQualityError(msg)
            elif self.global_action == QualityAction.WARN and self.log_warnings:
                logger.warning(f"{user_id[:8]} | QUALITY | global | {msg}")
            elif self.global_action == QualityAction.ERROR:
                raise ValueError(msg)
            
            return False
        return True
    
    def validate_te(self, n_samples: int, user_id: str, 
                   base_A: int = None, base_S: int = None, k: int = None) -> bool:
        """
        Validate Transfer Entropy sample requirements.
        
        Args:
            n_samples: Total sample count
            user_id: User identifier
            base_A: Activity alphabet size (optional, for state space check)
            base_S: Sitting alphabet size (optional)
            k: History length (optional)
            
        Returns:
            True if validation passed
            
        Raises:
            DataQualityError: If action=skip and threshold violated
        """
        # Basic sample count check
        if n_samples < self.te.min_samples:
            msg = f"TE samples {n_samples} < {self.te.min_samples}"
            return self._handle_violation('TE', msg, user_id, self.te.action)
        
        # State space check if parameters provided
        if self.te.min_samples_per_state and all([base_A, base_S, k]):
            state_space = (base_A ** k) * (base_S ** (k + 1))
            samples_per_state = n_samples / state_space
            
            if samples_per_state < self.te.min_samples_per_state:
                msg = f"TE samples/state {samples_per_state:.1f} < {self.te.min_samples_per_state}"
                return self._handle_violation('TE', msg, user_id, self.te.action)
        
        return True
    
    def validate_cte(self, n_samples: int, hour_counts: List[int], 
                    user_id: str) -> Tuple[bool, List[int], Dict]:
        """
        Validate CTE requirements and filter bins if needed.
        
        Args:
            n_samples: Total sample count
            hour_counts: Sample count per hour bin
            user_id: User identifier
            
        Returns:
            (passed, valid_bins, diagnostics)
            - passed: Whether CTE can proceed
            - valid_bins: List of valid bin indices
            - diagnostics: Quality metrics for reporting
        """
        diagnostics = {
            'total_bins': len(hour_counts),
            'low_bins': [],
            'valid_bins': [],
            'filtered_count': 0,
            'min_bin_count': min(hour_counts) if hour_counts else 0,
            'passed': True
        }
        
        # Total sample check
        if n_samples < self.cte.min_total_samples:
            msg = f"CTE total samples {n_samples} < {self.cte.min_total_samples}"
            diagnostics['passed'] = self._handle_violation('CTE', msg, user_id, 
                                                           self.cte.action)
            if not diagnostics['passed']:
                return False, [], diagnostics
        
        # Bin-level validation
        valid_bins = []
        low_bins = []
        
        for bin_idx, count in enumerate(hour_counts):
            if count < self.cte.min_bin_samples:
                low_bins.append(bin_idx)
            else:
                valid_bins.append(bin_idx)
        
        diagnostics['low_bins'] = low_bins
        diagnostics['valid_bins'] = valid_bins
        diagnostics['filtered_count'] = len(low_bins)
        filtered_ratio = len(low_bins) / len(hour_counts) if hour_counts else 0
        
        # Check filtering limits
        if len(low_bins) > self.cte.max_filtered_bins:
            msg = f"CTE filtered bins {len(low_bins)} > {self.cte.max_filtered_bins}"
            if self.log_warnings:
                logger.warning(f"{user_id[:8]} | QUALITY | CTE | {msg}")
        
        if filtered_ratio > self.cte.max_filtered_ratio:
            msg = f"CTE filtered ratio {filtered_ratio:.2%} > {self.cte.max_filtered_ratio:.2%}"
            if self.cte.action == QualityAction.SKIP:
                diagnostics['passed'] = False
                raise DataQualityError(msg)
            elif self.log_warnings:
                logger.warning(f"{user_id[:8]} | QUALITY | CTE | {msg}")
        
        # Apply action
        if self.cte.action == QualityAction.FILTER:
            if self.log_warnings and low_bins:
                logger.info(f"{user_id[:8]} | CTE_FILTER | kept={len(valid_bins)} "
                           f"filtered={len(low_bins)} ratio={filtered_ratio:.2%}")
            return True, valid_bins, diagnostics
        else:
            # Warn only, don't filter
            if self.log_warnings and low_bins:
                logger.warning(f"{user_id[:8]} | CTE_WARNING | low_bins={len(low_bins)}")
            return True, list(range(len(hour_counts))), diagnostics
    
    def validate_ste(self, n_samples: int, user_id: str) -> bool:
        """Validate Symbolic TE requirements."""
        if n_samples < self.ste.min_samples:
            msg = f"STE samples {n_samples} < {self.ste.min_samples}"
            return self._handle_violation('STE', msg, user_id, self.ste.action)
        return True
    
    def validate_gc(self, n_samples: int, user_id: str, max_lag: int = None) -> bool:
        """
        Validate Granger Causality requirements.
        
        Args:
            n_samples: Total sample count
            user_id: User identifier
            max_lag: Maximum lag for VAR model (optional)
        """
        if n_samples < self.gc.min_samples:
            msg = f"GC samples {n_samples} < {self.gc.min_samples}"
            return self._handle_violation('GC', msg, user_id, self.gc.action)
        
        # Check samples per lag if specified
        if self.gc.min_samples_per_state and max_lag:
            samples_per_lag = n_samples / max_lag
            if samples_per_lag < self.gc.min_samples_per_state:
                msg = f"GC samples/lag {samples_per_lag:.1f} < {self.gc.min_samples_per_state}"
                return self._handle_violation('GC', msg, user_id, self.gc.action)
        
        return True
    
    def validate_k_selection(self, n_samples: int, base: int, k: int, 
                            user_id: str) -> Tuple[bool, Optional[int]]:
        """
        Validate k-selection against undersampling.
        
        Args:
            n_samples: Total sample count
            base: Alphabet size
            k: Proposed history length
            user_id: User identifier
            
        Returns:
            (valid, fallback_k)
            - valid: Whether k is acceptable
            - fallback_k: Suggested k if current is invalid (or None)
        """
        state_space = base ** k
        samples_per_state = n_samples / state_space if state_space > 0 else 0
        
        if samples_per_state < self.k_selection.min_samples_per_state:
            msg = (f"k={k} undersampled: {samples_per_state:.1f} samples/state "
                   f"< {self.k_selection.min_samples_per_state}")
            
            if self.k_selection.enforce:
                if self.log_warnings:
                    logger.warning(f"{user_id[:8]} | K_SELECT | {msg} "
                                 f"→ fallback k={self.k_selection.fallback_k}")
                return False, self.k_selection.fallback_k
            else:
                if self.log_warnings:
                    logger.warning(f"{user_id[:8]} | K_SELECT | {msg} (not enforced)")
                return True, None
        
        # Check absolute max
        if k > self.k_selection.max_k_absolute:
            msg = f"k={k} exceeds max_k={self.k_selection.max_k_absolute}"
            if self.log_warnings:
                logger.warning(f"{user_id[:8]} | K_SELECT | {msg} → capped")
            return False, min(k, self.k_selection.max_k_absolute)
        
        return True, None
    
    def _handle_violation(self, method: str, message: str, user_id: str, 
                         action: QualityAction) -> bool:
        """
        Handle a quality threshold violation.
        
        Args:
            method: Analysis method name
            message: Violation description
            user_id: User identifier
            action: Action to take
            
        Returns:
            True if processing should continue, False otherwise
            
        Raises:
            DataQualityError: If action is SKIP
        """
        if action == QualityAction.SKIP:
            raise DataQualityError(f"{method}: {message}")
        elif action == QualityAction.ERROR:
            raise ValueError(f"{method}: {message}")
        elif action == QualityAction.WARN and self.log_warnings:
            logger.warning(f"{user_id[:8]} | QUALITY | {method} | {message}")
        
        return action == QualityAction.WARN
    
    def estimate_min_samples_te(self, base_A: int, base_S: int, k: int) -> int:
        """
        Estimate minimum required samples for TE analysis.
        
        Based on state space size and configured samples/state threshold.
        
        Args:
            base_A: Activity alphabet size
            base_S: Sitting alphabet size
            k: History length
            
        Returns:
            Estimated minimum sample count
        """
        state_space = (base_A ** k) * (base_S ** (k + 1))
        min_samples_per_state = self.te.min_samples_per_state or 5
        return int(state_space * min_samples_per_state)
    
    def generate_quality_summary(self, diagnostics: Dict) -> str:
        """
        Generate human-readable quality summary.
        
        Args:
            diagnostics: Quality check results
            
        Returns:
            Formatted summary string
        """
        lines = ["=== Data Quality Summary ==="]
        lines.append(f"Profile: {diagnostics.get('profile', 'unknown')}")
        lines.append(f"Total samples: {diagnostics.get('n_samples', 0)}")
        lines.append(f"Quality passed: {diagnostics.get('passed', False)}")
        
        if 'violations' in diagnostics:
            lines.append(f"Violations: {len(diagnostics['violations'])}")
            for v in diagnostics['violations']:
                lines.append(f"  - {v}")
        
        return "\n".join(lines)
