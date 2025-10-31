#!/usr/bin/env python
"""Proposal-aligned pipeline: CLEAN→ALIGN→SMOKE→RUN.

Implements research proposal specifications with best practices:
- Decoupled configuration
- AIS-based k selection per subject
- JIDT v1.5 native tau for TE/STE
- STRATIFIED-CTE with data-level lag
- Per-(family, tau) BH-FDR correction
- Checkpointing and heartbeat monitoring
"""
import sys
import json
import yaml
import logging
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import fdrcorrection

sys.path.insert(0, str(Path.cwd()))

from src.proposal_config import ProposalConfig, create_smoke_config, create_full_config
from src import settings
from src import preprocessing
from src import analysis
from src import granger_analysis
from src import symbolic_te
from src.k_selection import select_k_via_ais

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ProposalPipeline:
    """Proposal-aligned analysis pipeline."""
    
    def __init__(self, config: ProposalConfig):
        self.config = config
        self.out_dir = Path(config.output.out_dir)
        self.results = {
            'te': [],
            'cte': [],
            'ste': [],
            'gc': [],
            'k_selected': [],
            'errors': []
        }
        self.heartbeat_path = self.out_dir / config.output.heartbeat_file
    
    def setup(self):
        """Initialize output directories and JVM."""
        # Create directories
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'figs').mkdir(exist_ok=True)
        (self.out_dir / 'checkpoints').mkdir(exist_ok=True)
        
        # Save configuration
        with open(self.out_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)
        
        # Start JVM with configured heap
        analysis.start_jvm()
        
        # Log JIDT implementation details
        logger.warning(f"TE: JIDT v1.5 6-arg initialise(base, k_dest, 1, k_src, 1, delay)")
        logger.warning(f"CTE: STRATIFIED-TE with data-level lag before stratification")
        logger.warning(f"STE: JIDT v1.5 6-arg initialise")
        logger.warning(f"K selection: AIS over k∈{self.config.methods.k_range}")
    
    def update_heartbeat(self, status: str, **kwargs):
        """Update heartbeat status file."""
        heartbeat = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            **kwargs
        }
        with open(self.heartbeat_path, 'w') as f:
            json.dump(heartbeat, f, indent=2)
    
    def select_k_for_subject(self, uuid: str, S: np.ndarray, base_S: int) -> int:
        """Select k via AIS for destination (sleep) variable."""
        try:
            k_info = select_k_via_ais(
                S,
                base_S,
                self.config.methods.k_range,
                num_surrogates=100,  # Reduced for k selection
                criterion='max_ais'
            )
            k_selected = k_info['k_selected']
            
            self.results['k_selected'].append({
                'uuid': uuid,
                'k_selected': k_selected,
                'ais_values': k_info['ais_values'],
                'criterion': k_info['criterion']
            })
            
            logger.info(f"{uuid}: AIS selected k={k_selected}")
            return k_selected
        
        except Exception as e:
            logger.error(f"{uuid}: K selection failed: {e}")
            self.results['errors'].append({
                'uuid': uuid,
                'method': 'K_SELECTION',
                'error': str(e)[:200]
            })
            return 4  # Fallback default
    
    def compute_te(self, uuid: str, feat_mode: str, A: np.ndarray, S: np.ndarray, 
                   k: int, base_A: int, base_S: int):
        """Compute TE for all tau values with native JIDT tau."""
        for tau in self.config.methods.te_tau_values:
            try:
                te_result = analysis.run_te_analysis(
                    A.astype(int), S.astype(int),
                    k, k, base_A, base_S, tau=tau
                )
                
                self.results['te'].append({
                    'uuid': uuid,
                    'feature_mode': feat_mode,
                    'k': k,
                    'tau': tau,
                    'TE_A2S': te_result.get('TE(A->S)'),
                    'TE_S2A': te_result.get('TE(S->A)'),
                    'Delta_TE': te_result.get('Delta_TE'),
                    'p_A2S': te_result.get('p(A->S)'),
                    'p_S2A': te_result.get('p(S->A)')
                })
                gc.collect()
                
            except Exception as e:
                self.results['errors'].append({
                    'uuid': uuid, 'feature_mode': feat_mode, 'method': 'TE',
                    'tau': tau, 'error': str(e)[:200]
                })
    
    def compute_cte(self, uuid: str, feat_mode: str, A: np.ndarray, S: np.ndarray,
                    H_raw: np.ndarray, k: int, base_A: int, base_S: int):
        """Compute CTE with 24-hour bins using STRATIFIED-TE with data-level lag."""
        base_H = self.config.methods.cte_hour_bins
        
        for tau in self.config.methods.cte_tau_values:
            try:
                cte_result = analysis.run_cte_analysis(
                    A.astype(int), S.astype(int), H_raw,
                    k, k, base_A, base_S, base_H, tau=tau
                )
                
                self.results['cte'].append({
                    'uuid': uuid,
                    'feature_mode': feat_mode,
                    'k': k,
                    'tau': tau,
                    'CTE_A2S': cte_result.get('CTE(A->S|H_bin)'),
                    'CTE_S2A': cte_result.get('CTE(S->A|H_bin)'),
                    'Delta_CTE': cte_result.get('Delta_CTE_bin'),
                    'p_A2S': cte_result.get('p_cte(A->S|H_bin)'),
                    'p_S2A': cte_result.get('p_cte(S->A|H_bin)'),
                    'hour_bins_used': base_H
                })
                gc.collect()
                
            except Exception as e:
                self.results['errors'].append({
                    'uuid': uuid, 'feature_mode': feat_mode, 'method': 'CTE',
                    'tau': tau, 'error': str(e)[:200]
                })
    
    def compute_ste(self, uuid: str, feat_mode: str, A: np.ndarray, S: np.ndarray, k: int):
        """Compute STE with native JIDT tau."""
        for tau in self.config.methods.ste_tau_values:
            try:
                ste_result = symbolic_te.run_symbolic_te_analysis(
                    A, S, k, k, tau=tau
                )
                
                self.results['ste'].append({
                    'uuid': uuid,
                    'feature_mode': feat_mode,
                    'k': k,
                    'tau': tau,
                    'STE_A2S': ste_result.get('STE(A->S)'),
                    'STE_S2A': ste_result.get('STE(S->A)'),
                    'Delta_STE': ste_result.get('Delta_STE'),
                    'p_A2S': ste_result.get('p_ste(A->S)'),
                    'p_S2A': ste_result.get('p_ste(S->A)')
                })
                gc.collect()
                
            except Exception as e:
                self.results['errors'].append({
                    'uuid': uuid, 'feature_mode': feat_mode, 'method': 'STE',
                    'tau': tau, 'error': str(e)[:200]
                })
    
    def compute_gc(self, uuid: str, feat_mode: str, A: np.ndarray, S: np.ndarray):
        """Compute Granger causality (no tau parameter)."""
        try:
            gc_result = granger_analysis.run_granger_causality(A, S, max_lag=8)
            
            self.results['gc'].append({
                'uuid': uuid,
                'feature_mode': feat_mode,
                'GC_A2S_pval': gc_result.get('GC_A2S_pval'),
                'GC_S2A_pval': gc_result.get('GC_S2A_pval')
            })
            gc.collect()
            
        except Exception as e:
            self.results['errors'].append({
                'uuid': uuid, 'feature_mode': feat_mode, 'method': 'GC',
                'error': str(e)[:200]
            })
    
    def process_subject(self, uuid: str, feature_mode: str):
        """Process single subject with specified feature mode."""
        try:
            # Load and prepare data
            raw_data = preprocessing.load_subject_data(uuid)
            A, S, H_raw, H_bin = preprocessing.create_variables(raw_data, feature_mode=feature_mode)
            
            if len(A) == 0:
                self.results['errors'].append({
                    'uuid': uuid, 'feature_mode': feature_mode, 'error': 'empty_data'
                })
                return
            
            base_A = int(np.max(A)) + 1
            base_S = int(np.max(S)) + 1
            
            # Select k via AIS (only once per subject, reuse for all modes)
            k_selected = self.select_k_for_subject(uuid, S.astype(int), base_S)
            
            # Compute all methods
            self.compute_te(uuid, feature_mode, A, S, k_selected, base_A, base_S)
            self.compute_cte(uuid, feature_mode, A, S, H_raw, k_selected, base_A, base_S)
            self.compute_ste(uuid, feature_mode, A, S, k_selected)
            self.compute_gc(uuid, feature_mode, A, S)
            
            # Checkpoint
            checkpoint_file = self.out_dir / 'checkpoints' / f'{uuid}_{feature_mode}.json'
            checkpoint_file.write_text(json.dumps({
                'uuid': uuid,
                'feature_mode': feature_mode,
                'k_selected': k_selected,
                'completed': datetime.now().isoformat()
            }))
            
        except Exception as e:
            self.results['errors'].append({
                'uuid': uuid,
                'feature_mode': feature_mode,
                'method': 'PROCESS',
                'error': str(e)[:200]
            })
    
    def apply_fdr_correction(self, df: pd.DataFrame, p_col: str, q_col: str, family: str):
        """Apply BH-FDR correction per (family, tau)."""
        if p_col not in df.columns or len(df) == 0:
            return df
        
        df[q_col] = np.nan
        
        for tau_val in self.config.methods.te_tau_values:
            mask = (df['tau'] == tau_val) & df[p_col].notna()
            if mask.sum() > 0:
                _, q_vals = fdrcorrection(df.loc[mask, p_col].values, alpha=self.config.statistics.alpha)
                df.loc[mask, q_col] = q_vals
                logger.info(f"FDR {family} tau={tau_val}: {mask.sum()} tests, {(q_vals < 0.05).sum()} significant")
        
        return df
    
    def save_results(self):
        """Save all results with FDR correction."""
        # Convert to DataFrames
        df_te = pd.DataFrame(self.results['te'])
        df_cte = pd.DataFrame(self.results['cte'])
        df_ste = pd.DataFrame(self.results['ste'])
        df_gc = pd.DataFrame(self.results['gc'])
        df_k = pd.DataFrame(self.results['k_selected'])
        
        # Apply FDR correction per (family, tau)
        if len(df_te) > 0:
            df_te = self.apply_fdr_correction(df_te, 'p_A2S', 'q_A2S', 'TE')
        if len(df_cte) > 0:
            df_cte = self.apply_fdr_correction(df_cte, 'p_A2S', 'q_A2S', 'CTE')
        if len(df_ste) > 0:
            df_ste = self.apply_fdr_correction(df_ste, 'p_A2S', 'q_A2S', 'STE')
        
        # Save files
        for fmt in self.config.output.save_formats:
            if fmt == 'csv':
                df_te.to_csv(self.out_dir / 'per_user_te.csv', index=False)
                df_cte.to_csv(self.out_dir / 'per_user_cte.csv', index=False)
                df_ste.to_csv(self.out_dir / 'per_user_ste.csv', index=False)
                df_gc.to_csv(self.out_dir / 'per_user_gc.csv', index=False)
                df_k.to_csv(self.out_dir / 'k_selected_by_user.csv', index=False)
            elif fmt == 'parquet':
                if len(df_te) > 0:
                    df_te.to_parquet(self.out_dir / 'per_user_te.parquet', index=False)
                if len(df_cte) > 0:
                    df_cte.to_parquet(self.out_dir / 'per_user_cte.parquet', index=False)
                if len(df_ste) > 0:
                    df_ste.to_parquet(self.out_dir / 'per_user_ste.parquet', index=False)
        
        # Sign audit
        self.create_sign_audit(df_te, df_cte, df_ste)
        
        # FDR report
        self.create_fdr_report(df_te, df_cte, df_ste)
        
        # Error log
        if self.results['errors']:
            pd.DataFrame(self.results['errors']).to_csv(self.out_dir / 'error_log.csv', index=False)
        
        # Run info
        self.save_run_info()
        
        return df_te, df_cte, df_ste, df_gc, df_k
    
    def create_sign_audit(self, df_te, df_cte, df_ste):
        """Create cross-method sign consistency audit."""
        sign_audit = []
        
        all_uuids = set(df_te['uuid'].unique()) | set(df_cte['uuid'].unique()) | set(df_ste['uuid'].unique())
        
        for uuid in all_uuids:
            for feat_mode in self.config.feature_modes:
                for tau in self.config.methods.te_tau_values:
                    te_delta = df_te[(df_te['uuid']==uuid) & (df_te['feature_mode']==feat_mode) & (df_te['tau']==tau)]['Delta_TE'].median() if len(df_te) > 0 else np.nan
                    cte_delta = df_cte[(df_cte['uuid']==uuid) & (df_cte['feature_mode']==feat_mode) & (df_cte['tau']==tau)]['Delta_CTE'].median() if len(df_cte) > 0 else np.nan
                    ste_delta = df_ste[(df_ste['uuid']==uuid) & (df_ste['feature_mode']==feat_mode) & (df_ste['tau']==tau)]['Delta_STE'].median() if len(df_ste) > 0 else np.nan
                    
                    signs = []
                    if pd.notna(te_delta): signs.append('TE+' if te_delta > 0 else 'TE-')
                    if pd.notna(cte_delta): signs.append('CTE+' if cte_delta > 0 else 'CTE-')
                    if pd.notna(ste_delta): signs.append('STE+' if ste_delta > 0 else 'STE-')
                    
                    sign_audit.append({
                        'uuid': uuid,
                        'feature_mode': feat_mode,
                        'tau': tau,
                        'signs': ','.join(signs),
                        'all_agree': len(set([s[-1] for s in signs])) == 1 if signs else False
                    })
        
        pd.DataFrame(sign_audit).to_csv(self.out_dir / 'sign_audit.csv', index=False)
    
    def create_fdr_report(self, df_te, df_cte, df_ste):
        """Create FDR correction report."""
        fdr_rows = []
        
        for tau in self.config.methods.te_tau_values:
            for family, df, p_col, q_col in [
                ('TE', df_te, 'p_A2S', 'q_A2S'),
                ('CTE', df_cte, 'p_A2S', 'q_A2S'),
                ('STE', df_ste, 'p_A2S', 'q_A2S')
            ]:
                if len(df) == 0:
                    continue
                
                df_tau = df[df['tau'] == tau]
                if len(df_tau) == 0:
                    continue
                
                fdr_rows.append({
                    'family': family,
                    'tau': tau,
                    'n_tests': len(df_tau),
                    'n_sig_p': (df_tau[p_col] < 0.05).sum() if p_col in df_tau.columns else 0,
                    'n_sig_q': (df_tau[q_col] < 0.05).sum() if q_col in df_tau.columns else 0
                })
        
        pd.DataFrame(fdr_rows).to_csv(self.out_dir / 'fdr_report.csv', index=False)
    
    def save_run_info(self):
        """Save run metadata."""
        with open(self.out_dir / 'run_info.yaml', 'w') as f:
            yaml.dump({
                'timestamp': datetime.now().isoformat(),
                'config': self.config.to_dict(),
                'n_users_processed': len(set([r['uuid'] for r in self.results['te']])),
                'n_errors': len(self.results['errors'])
            }, f, default_flow_style=False)
    
    def run(self, user_list: List[str]):
        """Run pipeline for specified users."""
        self.update_heartbeat('running', users_total=len(user_list))
        
        for i, uuid in enumerate(tqdm(user_list, desc='Users')):
            for feat_mode in self.config.feature_modes:
                self.process_subject(uuid, feat_mode)
                self.update_heartbeat(
                    'running',
                    users_completed=i+1,
                    users_total=len(user_list),
                    current_user=uuid,
                    current_mode=feat_mode
                )
        
        df_te, df_cte, df_ste, df_gc, df_k = self.save_results()
        self.update_heartbeat('completed')
        
        return self.generate_report(df_te, df_cte, df_ste, df_gc, df_k)
    
    def generate_report(self, df_te, df_cte, df_ste, df_gc, df_k) -> Dict[str, Any]:
        """Generate final one-line JSON report."""
        group_results = {}
        
        for tau in self.config.methods.te_tau_values:
            te_tau = df_te[df_te['tau'] == tau] if len(df_te) > 0 else pd.DataFrame()
            cte_tau = df_cte[df_cte['tau'] == tau] if len(df_cte) > 0 else pd.DataFrame()
            ste_tau = df_ste[df_ste['tau'] == tau] if len(df_ste) > 0 else pd.DataFrame()
            
            group_results[f'tau{tau}'] = {
                'TE': {
                    'median': float(te_tau['Delta_TE'].median()) if len(te_tau) > 0 else np.nan,
                    'CI95': te_tau['Delta_TE'].quantile([0.025, 0.975]).tolist() if len(te_tau) > 0 else [np.nan, np.nan],
                    'q_sig_rate': float((te_tau['q_A2S'] < 0.05).mean()) if 'q_A2S' in te_tau.columns and len(te_tau) > 0 else np.nan
                },
                'CTE24h': {
                    'median': float(cte_tau['Delta_CTE'].median()) if len(cte_tau) > 0 else np.nan,
                    'CI95': cte_tau['Delta_CTE'].quantile([0.025, 0.975]).tolist() if len(cte_tau) > 0 else [np.nan, np.nan],
                    'q_sig_rate': float((cte_tau['q_A2S'] < 0.05).mean()) if 'q_A2S' in cte_tau.columns and len(cte_tau) > 0 else np.nan
                },
                'STE': {
                    'median': float(ste_tau['Delta_STE'].median()) if len(ste_tau) > 0 else np.nan,
                    'CI95': ste_tau['Delta_STE'].quantile([0.025, 0.975]).tolist() if len(ste_tau) > 0 else [np.nan, np.nan],
                    'q_sig_rate': float((ste_tau['q_A2S'] < 0.05).mean()) if 'q_A2S' in ste_tau.columns and len(ste_tau) > 0 else np.nan
                }
            }
        
        return {
            'status': 'done',
            'OUT_DIR': str(self.out_dir.resolve()),
            'coverage': {
                'users': len(df_k) if len(df_k) > 0 else 0,
                'modes': len(self.config.feature_modes)
            },
            'params': {
                'hour_bins': self.config.methods.cte_hour_bins,
                'taus': self.config.methods.te_tau_values,
                'k_strategy': self.config.methods.k_selection_method,
                'k_range': self.config.methods.k_range
            },
            'group_results': group_results,
            'GC_agree_rate': float((df_gc['GC_A2S_pval'] < 0.05).mean()) if len(df_gc) > 0 else np.nan,
            'artifacts': {
                'per_user_te': str((self.out_dir / 'per_user_te.csv').resolve()),
                'per_user_cte': str((self.out_dir / 'per_user_cte.csv').resolve()),
                'per_user_ste': str((self.out_dir / 'per_user_ste.csv').resolve()),
                'per_user_gc': str((self.out_dir / 'per_user_gc.csv').resolve()),
                'sign_audit': str((self.out_dir / 'sign_audit.csv').resolve()),
                'k_selected_by_user': str((self.out_dir / 'k_selected_by_user.csv').resolve()),
                'fdr_report': str((self.out_dir / 'fdr_report.csv').resolve()),
                'run_info': str((self.out_dir / 'run_info.yaml').resolve())
            }
        }


def main_smoke():
    """SMOKE test: 2 users, composite only."""
    import glob
    
    config = create_smoke_config()
    pipeline = ProposalPipeline(config)
    pipeline.setup()
    
    # Get first 2 users
    data_root = Path('data')
    subject_files = glob.glob(str(data_root / 'ExtraSensory.per_uuid_features_labels' / '*.features_labels.csv'))
    all_uuids = sorted([Path(f).stem.replace('.features_labels', '') for f in subject_files])[:2]
    
    logger.warning(f"SMOKE TEST: {len(all_uuids)} users, modes={config.feature_modes}")
    
    result = pipeline.run(all_uuids)
    print(json.dumps(result, separators=(',', ':')))
    
    analysis.shutdown_jvm()
    return result


def main_full():
    """RUN: Full 60 users, all modes."""
    import glob
    
    config = create_full_config()
    pipeline = ProposalPipeline(config)
    pipeline.setup()
    
    # Get all 60 users
    data_root = Path('data')
    subject_files = glob.glob(str(data_root / 'ExtraSensory.per_uuid_features_labels' / '*.features_labels.csv'))
    all_uuids = sorted([Path(f).stem.replace('.features_labels', '') for f in subject_files])[:60]
    
    logger.warning(f"FULL RUN: {len(all_uuids)} users, modes={config.feature_modes}")
    
    result = pipeline.run(all_uuids)
    print(json.dumps(result, separators=(',', ':')))
    
    analysis.shutdown_jvm()
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        main_full()
    else:
        main_smoke()
