# src/quality_control.py
# Data quality validation and threshold enforcement
# Centralizes all sample size requirements and quality checks

import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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
    
    def estimate_min_samples_ais(self, base: int, k: int) -> int:
        """
        Estimate minimum required samples for AIS k-selection.
        
        Args:
            base: Alphabet size
            k: History length
            
        Returns:
            Estimated minimum sample count
        """
        state_space = base ** k
        return int(state_space * self.k_selection.min_samples_per_state)
    
    def recommend_max_k(self, n_samples: int, base: int) -> int:
        """
        Recommend maximum k based on available samples.
        
        Ensures minimum samples/state threshold is met.
        
        Args:
            n_samples: Available sample count
            base: Alphabet size
            
        Returns:
            Recommended maximum k
        """
        min_sps = self.k_selection.min_samples_per_state
        
        # Find largest k where base^k * min_sps <= n_samples
        k_max = 1
        while True:
            state_space = base ** (k_max + 1)
            required_samples = state_space * min_sps
            if required_samples > n_samples:
                break
            k_max += 1
            if k_max >= self.k_selection.max_k_absolute:
                break
        
        return min(k_max, self.k_selection.max_k_absolute)
    
    def estimate_statistical_power(self, n_samples: int, state_space: int, 
                                   alpha: float = 0.05) -> float:
        """
        Estimate statistical power for information-theoretic test.
        
        Simplified power estimation based on sample size and state space.
        Assumes chi-square test with df = state_space - 1.
        
        Args:
            n_samples: Sample count
            state_space: Total number of states
            alpha: Significance level
            
        Returns:
            Estimated power (0.0-1.0)
        """
        samples_per_state = n_samples / state_space if state_space > 0 else 0
        
        # Empirical power curve based on Lizier (2014) recommendations
        # Power ~0.50 at 2 samples/state, ~0.80 at 10 samples/state
        if samples_per_state < 1:
            return 0.20
        elif samples_per_state < 2:
            return 0.50
        elif samples_per_state < 5:
            return 0.70
        elif samples_per_state < 10:
            return 0.80
        else:
            return min(0.95, 0.80 + (samples_per_state - 10) * 0.01)
    
    def get_quality_diagnostics(self, n_samples: int, base_A: int = None, 
                               base_S: int = None, k: int = None) -> Dict:
        """
        Generate comprehensive quality diagnostics for dataset.
        
        Args:
            n_samples: Total sample count
            base_A: Activity alphabet size (optional)
            base_S: Sitting alphabet size (optional)
            k: History length (optional)
            
        Returns:
            Dictionary with quality metrics and recommendations
        """
        diagnostics = {
            'n_samples': n_samples,
            'global_threshold': self.min_total_samples,
            'global_passed': n_samples >= self.min_total_samples,
            'recommendations': []
        }
        
        # TE analysis diagnostics
        if base_A and base_S and k:
            te_state_space = (base_A ** k) * (base_S ** (k + 1))
            te_sps = n_samples / te_state_space
            te_min_samples = self.estimate_min_samples_te(base_A, base_S, k)
            te_power = self.estimate_statistical_power(n_samples, te_state_space)
            
            diagnostics['te'] = {
                'state_space': te_state_space,
                'samples_per_state': te_sps,
                'min_samples_required': te_min_samples,
                'threshold_met': n_samples >= te_min_samples,
                'estimated_power': te_power,
                'power_level': 'high' if te_power >= 0.80 else 'medium' if te_power >= 0.70 else 'low'
            }
            
            if te_power < 0.70:
                diagnostics['recommendations'].append(
                    f"TE power={te_power:.2f} is low. Consider reducing k or increasing sample size."
                )
        
        # k-selection diagnostics
        if base_A and k:
            ais_state_space = base_A ** k
            ais_sps = n_samples / ais_state_space
            ais_min_samples = self.estimate_min_samples_ais(base_A, k)
            rec_max_k = self.recommend_max_k(n_samples, base_A)
            
            diagnostics['k_selection'] = {
                'state_space': ais_state_space,
                'samples_per_state': ais_sps,
                'min_samples_required': ais_min_samples,
                'threshold_met': n_samples >= ais_min_samples,
                'recommended_max_k': rec_max_k,
                'undersampled': k > rec_max_k
            }
            
            if k > rec_max_k:
                diagnostics['recommendations'].append(
                    f"k={k} may be undersampled. Recommended max k={rec_max_k} for {n_samples} samples."
                )
        
        # CTE diagnostics
        diagnostics['cte'] = {
            'min_total_samples': self.cte.min_total_samples,
            'min_bin_samples': self.cte.min_bin_samples,
            'threshold_met': n_samples >= self.cte.min_total_samples
        }
        
        return diagnostics
    
    def generate_quality_summary(self, diagnostics: Dict) -> str:
        """
        Generate human-readable quality summary.
        
        Args:
            diagnostics: Quality check results from get_quality_diagnostics()
            
        Returns:
            Formatted summary string
        """
        lines = ["=" * 60]
        lines.append("Data Quality Summary")
        lines.append("=" * 60)
        
        # Global metrics
        lines.append(f"\nGlobal:")
        lines.append(f"  Samples: {diagnostics['n_samples']:,}")
        lines.append(f"  Threshold: {diagnostics['global_threshold']:,}")
        lines.append(f"  Status: {'✓ PASS' if diagnostics['global_passed'] else '✗ FAIL'}")
        
        # TE diagnostics
        if 'te' in diagnostics:
            te = diagnostics['te']
            lines.append(f"\nTransfer Entropy:")
            lines.append(f"  State space: {te['state_space']:,}")
            lines.append(f"  Samples/state: {te['samples_per_state']:.2f}")
            lines.append(f"  Min required: {te['min_samples_required']:,}")
            lines.append(f"  Threshold: {'✓ MET' if te['threshold_met'] else '✗ NOT MET'}")
            lines.append(f"  Est. power: {te['estimated_power']:.2f} ({te['power_level']})")
        
        # k-selection diagnostics
        if 'k_selection' in diagnostics:
            ks = diagnostics['k_selection']
            lines.append(f"\nk-Selection (AIS):")
            lines.append(f"  State space: {ks['state_space']:,}")
            lines.append(f"  Samples/state: {ks['samples_per_state']:.2f}")
            lines.append(f"  Min required: {ks['min_samples_required']:,}")
            lines.append(f"  Threshold: {'✓ MET' if ks['threshold_met'] else '✗ NOT MET'}")
            lines.append(f"  Recommended max k: {ks['recommended_max_k']}")
            if ks['undersampled']:
                lines.append(f"  Warning: Current k is undersampled!")
        
        # CTE diagnostics
        if 'cte' in diagnostics:
            cte = diagnostics['cte']
            lines.append(f"\nConditional Transfer Entropy:")
            lines.append(f"  Min total samples: {cte['min_total_samples']:,}")
            lines.append(f"  Min bin samples: {cte['min_bin_samples']:,}")
            lines.append(f"  Threshold: {'✓ MET' if cte['threshold_met'] else '✗ NOT MET'}")
        
        # Recommendations
        if diagnostics.get('recommendations'):
            lines.append(f"\nRecommendations:")
            for rec in diagnostics['recommendations']:
                lines.append(f"  • {rec}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def generate_quality_report(self, results: Dict, out_dir: Path) -> Path:
        """
        Generate comprehensive quality report from pipeline results.
        
        Args:
            results: Pipeline results dictionary with quality metrics
            out_dir: Output directory for report
            
        Returns:
            Path to generated report file
        """
        import pandas as pd
        from datetime import datetime
        
        report_path = out_dir / 'quality_report.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# Quality Control Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Profile**: {results.get('profile', 'unknown')}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            
            if 'te' in results:
                te_df = pd.DataFrame(results['te'])
                if not te_df.empty:
                    total_te = len(te_df)
                    passed_te = te_df['quality_passed'].sum()
                    low_n_te = te_df['low_n'].sum()
                    
                    f.write(f"### Transfer Entropy (TE)\n\n")
                    f.write(f"- Total analyses: {total_te}\n")
                    f.write(f"- Quality passed: {passed_te} ({100*passed_te/total_te:.1f}%)\n")
                    f.write(f"- Low sample warnings: {low_n_te} ({100*low_n_te/total_te:.1f}%)\n")
                    f.write(f"- Mean samples: {te_df['n_samples'].mean():.0f}\n")
                    f.write(f"- Median samples: {te_df['n_samples'].median():.0f}\n\n")
            
            if 'cte' in results:
                cte_df = pd.DataFrame(results['cte'])
                if not cte_df.empty:
                    total_cte = len(cte_df)
                    passed_cte = cte_df['quality_passed'].sum()
                    bins_filtered = cte_df['bins_filtered'].sum()
                    
                    f.write(f"### Conditional Transfer Entropy (CTE)\n\n")
                    f.write(f"- Total analyses: {total_cte}\n")
                    f.write(f"- Quality passed: {passed_cte} ({100*passed_cte/total_cte:.1f}%)\n")
                    f.write(f"- Total bins filtered: {bins_filtered}\n")
                    f.write(f"- Mean bins/analysis: {cte_df['bins_filtered'].mean():.1f}\n")
                    f.write(f"- Mean samples: {cte_df['n_samples'].mean():.0f}\n\n")
            
            if 'ste' in results:
                ste_df = pd.DataFrame(results['ste'])
                if not ste_df.empty:
                    total_ste = len(ste_df)
                    passed_ste = ste_df['quality_passed'].sum()
                    
                    f.write(f"### Symbolic Transfer Entropy (STE)\n\n")
                    f.write(f"- Total analyses: {total_ste}\n")
                    f.write(f"- Quality passed: {passed_ste} ({100*passed_ste/total_ste:.1f}%)\n")
                    f.write(f"- Mean samples: {ste_df['n_samples'].mean():.0f}\n\n")
            
            if 'gc' in results:
                gc_df = pd.DataFrame(results['gc'])
                if not gc_df.empty:
                    total_gc = len(gc_df)
                    passed_gc = gc_df['quality_passed'].sum()
                    
                    f.write(f"### Granger Causality (GC)\n\n")
                    f.write(f"- Total analyses: {total_gc}\n")
                    f.write(f"- Quality passed: {passed_gc} ({100*passed_gc/total_gc:.1f}%)\n")
                    f.write(f"- Mean samples: {gc_df['n_samples'].mean():.0f}\n\n")
            
            # Quality violations
            f.write("## Quality Violations\n\n")
            
            violations = []
            if 'te' in results:
                te_df = pd.DataFrame(results['te'])
                if not te_df.empty:
                    failed_te = te_df[~te_df['quality_passed']]
                    if len(failed_te) > 0:
                        violations.append(f"- TE: {len(failed_te)} analyses failed quality checks")
            
            if 'cte' in results:
                cte_df = pd.DataFrame(results['cte'])
                if not cte_df.empty:
                    failed_cte = cte_df[~cte_df['quality_passed']]
                    if len(failed_cte) > 0:
                        violations.append(f"- CTE: {len(failed_cte)} analyses failed quality checks")
            
            if violations:
                for v in violations:
                    f.write(f"{v}\n")
            else:
                f.write("No quality violations detected.\n")
            
            f.write(f"\n---\n")
            f.write(f"*Report generated by ExtraSensory Quality Control System*\n")
        
        logger.info(f"Quality report saved to {report_path}")
        return report_path
