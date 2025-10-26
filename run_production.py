#!/usr/bin/env python
"""Production-ready pipeline with tracking, heartbeat, and monitoring.

Features:
- run_info.yaml (JIDT version, JVM params, git commit, seed)
- k_selected_by_user.csv (AIS k-selection tracking)
- hbin_counts.csv (configurable hour bins, typically 6 or 24)
- status.json (continuous heartbeat with ETA)
- errors.log (real-time error logging)
- CTE hour_bins from config, low_n_hours preserved
"""
import sys, json, logging, glob, gc, yaml, subprocess, time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))

from src import preprocessing, analysis, granger_analysis, symbolic_te
from src.fdr_utils import apply_fdr_per_family_tau, compute_delta_pvalue
from src.k_selection import select_k_via_ais

# Setup logging to both file and console
def setup_logging(out_dir):
    """Configure logging to errors.log and console."""
    log_file = out_dir / 'errors.log'
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

logger = logging.getLogger(__name__)


class ProductionPipeline:
    """Production pipeline with comprehensive tracking and monitoring."""
    
    def __init__(self, config_path, resume_dir=None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Validate required config fields
        required_fields = ['hour_bins', 'taus', 'k_selection', 'surrogates', 'fdr', 'feature_modes', 'out_dir']
        missing = [f for f in required_fields if f not in self.config]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")
        
        # Validate hour_bins
        if not isinstance(self.config['hour_bins'], int) or self.config['hour_bins'] < 1:
            raise ValueError(f"hour_bins must be int >= 1, got {self.config['hour_bins']}")
        
        # Create or resume output directory
        if resume_dir:
            self.out_dir = Path(resume_dir)
            if not self.out_dir.exists():
                raise ValueError(f"Resume directory does not exist: {resume_dir}")
            self.is_resume = True
            logger.warning(f"RESUME MODE: Continuing from {self.out_dir}")
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M')
            out_base = self.config['out_dir'].replace('<STAMP>', ts)
            self.out_dir = Path(out_base)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.is_resume = False
        
        # Setup logging
        setup_logging(self.out_dir)
        
        # Tracking
        self.results = {'te': [], 'cte': [], 'ste': [], 'gc': [], 'k_selected': [], 'hbin_counts': []}
        self.errors = []
        self.start_time = None
        self.users_completed = 0
        self.total_users = 0
        self.completed_combinations = set()  # Track (user_id, feature_mode) combinations
    
    def get_git_info(self):
        """Get git commit hash and status."""
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            status = subprocess.check_output(['git', 'status', '--short'], text=True).strip()
            return {
                'commit': commit[:8],
                'dirty': len(status) > 0,
                'status': status if len(status) > 0 else 'clean'
            }
        except:
            return {'commit': 'unknown', 'dirty': False, 'status': 'N/A'}
    
    def get_jidt_version(self):
        """Extract JIDT version from jar."""
        jar_path = Path('jidt/infodynamics.jar')
        if jar_path.exists():
            return {'jar': str(jar_path), 'version': 'v1.5', 'detected': True}
        return {'jar': 'N/A', 'version': 'unknown', 'detected': False}
    
    def load_checkpoint(self):
        """Load existing results from checkpoint files."""
        checkpoint_files = {
            'te': self.out_dir / 'per_user_te.csv',
            'cte': self.out_dir / 'per_user_cte.csv',
            'ste': self.out_dir / 'per_user_ste.csv',
            'gc': self.out_dir / 'per_user_gc.csv',
            'k_selected': self.out_dir / 'k_selected_by_user.csv',
            'hbin_counts': self.out_dir / 'hbin_counts.csv'
        }
        
        for key, fpath in checkpoint_files.items():
            if fpath.exists():
                df = pd.read_csv(fpath)
                logger.warning(f"CHECKPOINT: Loaded {len(df)} rows from {fpath.name}")
                
                # Track completed (user_id, feature_mode) combinations
                if 'user_id' in df.columns and 'feature_mode' in df.columns:
                    for _, row in df.iterrows():
                        self.completed_combinations.add((row['user_id'], row['feature_mode']))
                elif 'user_id' in df.columns:
                    # For k_selected and hbin_counts (no feature_mode)
                    for _, row in df.iterrows():
                        # Mark all feature modes as having k selected
                        for mode in self.config.get('feature_modes', ['composite']):
                            self.completed_combinations.add((row['user_id'], mode))
        
        logger.warning(f"CHECKPOINT: {len(self.completed_combinations)} combinations already completed")
        return len(self.completed_combinations)
    
    def save_checkpoint(self, method, data):
        """Save incremental checkpoint after each user-method completion."""
        file_map = {
            'te': 'per_user_te.csv',
            'cte': 'per_user_cte.csv',
            'ste': 'per_user_ste.csv',
            'gc': 'per_user_gc.csv',
            'k_selected': 'k_selected_by_user.csv',
            'hbin_counts': 'hbin_counts.csv'
        }
        
        if method not in file_map:
            return
        
        fpath = self.out_dir / file_map[method]
        df_new = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        
        # Append to existing file or create new
        if fpath.exists():
            df_new.to_csv(fpath, mode='a', header=False, index=False)
        else:
            df_new.to_csv(fpath, index=False)
    
    def write_run_info(self, seed=None):
        """Write run_info.yaml with environment details."""
        git_info = self.get_git_info()
        jidt_info = self.get_jidt_version()
        
        jvm_opts = self.config.get('jvm', {})
        
        run_info = {
            'timestamp': datetime.now().isoformat(),
            'schema_version': self.config.get('schema_version', 'v1.0'),
            'git': git_info,
            'jidt': jidt_info,
            'jvm': {
                'heap_min': jvm_opts.get('xms', '32g'),
                'heap_max': jvm_opts.get('xmx', '48g'),
                'gc': jvm_opts.get('opts', []),
                'classpath': 'jidt/infodynamics.jar'
            },
            'random_seed': seed if seed else 'system_default',
            'config': self.config,
            'implementation': {
                'TE': 'JIDT-v1.5-6arg-initialise(base,k_dest,1,k_src,1,delay)',
                'CTE': 'STRATIFIED-TE-with-data-level-lag-before-stratification',
                'STE': 'JIDT-v1.5-6arg-initialise',
                'GC': 'statsmodels-VAR-AIC',
                'FDR': 'BH-per-family-per-tau'
            }
        }
        
        with open(self.out_dir / 'run_info.yaml', 'w') as f:
            yaml.dump(run_info, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"run_info.yaml written: git={git_info['commit']}, JIDT={jidt_info['version']}")
    
    def update_heartbeat(self, current_user=None):
        """Update status.json with progress and ETA."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        if self.users_completed > 0:
            avg_time_per_user = elapsed / self.users_completed
            remaining_users = self.total_users - self.users_completed
            eta_seconds = avg_time_per_user * remaining_users
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)
            eta_str = eta_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            eta_str = 'calculating...'
        
        status = {
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'progress': {
                'done': self.users_completed,
                'total': self.total_users,
                'percent': round(100 * self.users_completed / self.total_users, 1) if self.total_users > 0 else 0
            },
            'current_user': current_user,
            'elapsed_seconds': int(elapsed),
            'elapsed_formatted': str(timedelta(seconds=int(elapsed))),
            'eta': eta_str,
            'errors_count': len(self.errors)
        }
        
        with open(self.out_dir / 'status.json', 'w') as f:
            json.dump(status, f, indent=2)
    
    def compute_hbin_counts(self, H_raw, user_id, feature_mode):
        """Compute per-hour bin sample counts."""
        if len(H_raw) == 0:
            return
        
        # Count samples per hour (0-23)
        bin_counts = {}
        for h in range(24):
            bin_counts[f'hour_{h:02d}'] = int((H_raw == h).sum())
        
        hbin_result = {
            'user_id': user_id,
            'feature_mode': feature_mode,
            'total_samples': len(H_raw),
            **bin_counts
        }
        self.results['hbin_counts'].append(hbin_result)
        self.save_checkpoint('hbin_counts', hbin_result)
    
    def select_k(self, S, base_S, user_id, H_raw=None):
        """Select k via AIS or fixed k.
        
        AIS strategy supports optional constraints:
        - k_max: Hard cap on k (e.g., 4 for computational feasibility)
        - undersampling_guard: Prevent undersampled states (min 25 samples/state)
        """
        k_strategy = self.config.get('k_selection', {}).get('strategy', 'fixed')
        
        if k_strategy == 'AIS':
            try:
                k_config = self.config['k_selection']
                k_grid = k_config['k_grid']
                k_max = k_config.get('k_max')  # Can be None for no limit
                undersampling_guard = k_config.get('undersampling_guard', False)
                
                min_samples = None
                if undersampling_guard and H_raw is not None:
                    min_samples = min([(H_raw == h).sum() for h in range(24)])
                
                k_info = select_k_via_ais(
                    S.astype(int), base_S, k_grid,
                    num_surrogates=100, criterion='max_ais',
                    k_max=k_max, min_samples=min_samples
                )
                k_selected = k_info['k_selected']
                
                k_result = {
                    'user_id': user_id,
                    'k_selected': k_selected,
                    'k_original': k_info.get('k_original', k_selected),
                    'capped': k_info.get('capped', False),
                    'ais_values': json.dumps(k_info['ais_values']),
                    'criterion': k_info['criterion']
                }
                self.results['k_selected'].append(k_result)
                self.save_checkpoint('k_selected', k_result)
                
                cap_msg = f" (capped from {k_info['k_original']})" if k_info.get('capped') else ""
                logger.info(f"{user_id}: AIS selected k={k_selected}{cap_msg}")
                return k_selected
            except Exception as e:
                logger.error(f"{user_id}: AIS k-selection failed: {e}")
                self.errors.append({'user_id': user_id, 'method': 'K_SELECTION', 'error': str(e)[:200]})
                return 4  # Fallback
        else:
            # Fixed k from config
            return self.config.get('k_selection', {}).get('k_fixed', 4)
    
    def process_user(self, user_id, feature_mode):
        """Process single user-feature combination."""
        # Check if already completed (resume mode)
        if (user_id, feature_mode) in self.completed_combinations:
            logger.warning(f"SKIP {user_id}/{feature_mode} (already completed)")
            return
        
        try:
            start_time = datetime.now()
            logger.warning(f"START {user_id}/{feature_mode} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            # Load data
            raw = preprocessing.load_subject_data(user_id)
            hour_bins = self.config['hour_bins']
            A, S, H_raw, H_binned = preprocessing.create_variables(raw, feature_mode=feature_mode, hour_bins=hour_bins)
            logger.warning(f"LOADED {user_id}/{feature_mode}: {len(A)} samples")
            
            if len(A) == 0:
                self.errors.append({'user_id': user_id, 'feature_mode': feature_mode, 'error': 'empty_data'})
                return
            
            base_A, base_S = int(np.max(A)) + 1, int(np.max(S)) + 1
            base_H = int(np.max(H_binned)) + 1
            logger.warning(f"BASES {user_id}: A={base_A}, S={base_S}, H_binned={base_H}")
            
            # Select k (only once per user, reuse for all modes)
            logger.warning(f"K-SELECT START {user_id}")
            k = l = self.select_k(S, base_S, user_id, H_raw=H_raw)
            logger.warning(f"K-SELECT DONE {user_id}: k={k}")
            
            # Compute hour bin counts (using H_binned)
            self.compute_hbin_counts(H_binned, user_id, feature_mode)
            
            # Get low_n_hours for CTE (using H_binned)
            num_bins = base_H
            hour_counts = [(H_binned == h).sum() for h in range(num_bins)]
            low_n_hours = [h for h, cnt in enumerate(hour_counts) if cnt < 50]
            
            # TE
            for tau in self.config['taus']:
                try:
                    logger.warning(f"TE tau={tau} START {user_id}/{feature_mode}")
                    te = analysis.run_te_analysis(A.astype(int), S.astype(int), k, l, base_A, base_S, tau=tau)
                    logger.warning(f"TE tau={tau} DONE {user_id}/{feature_mode}")
                    
                    te_result = {
                        'user_id': user_id, 'feature_mode': feature_mode, 'k': k, 'l': l, 'tau': tau,
                        'TE_A2S': te.get('TE(A->S)'), 'TE_S2A': te.get('TE(S->A)'),
                        'Delta_TE': te.get('Delta_TE'), 'p_A2S': te.get('p(A->S)'),
                        'p_S2A': te.get('p(S->A)', np.nan), 'n_samples': len(A), 'low_n': len(A) < 100
                    }
                    self.results['te'].append(te_result)
                    
                    # Save checkpoint after each TE completion
                    self.save_checkpoint('te', te_result)
                    gc.collect()
                except Exception as e:
                    self.errors.append({'user_id': user_id, 'feature_mode': feature_mode, 'method': 'TE', 'tau': tau, 'error': str(e)[:200]})
            
            # CTE
            for tau in self.config['taus']:
                try:
                    logger.warning(f"CTE tau={tau} START {user_id}/{feature_mode}")
                    cte = analysis.run_cte_analysis(A.astype(int), S.astype(int), H_binned, k, l, base_A, base_S, self.config['hour_bins'], tau=tau)
                    logger.warning(f"CTE tau={tau} DONE {user_id}/{feature_mode}")
                    
                    cte_result = {
                        'user_id': user_id, 'feature_mode': feature_mode, 'k': k, 'l': l, 'tau': tau,
                        'hour_bins': self.config['hour_bins'],  # From config (typically 6 or 24)
                        'CTE_A2S': cte.get('CTE(A->S|H_bin)'), 'CTE_S2A': cte.get('CTE(S->A|H_bin)'),
                        'Delta_CTE': cte.get('Delta_CTE_bin'), 'p_A2S': cte.get('p_cte(A->S|H_bin)'),
                        'p_S2A': cte.get('p_cte(S->A|H_bin)', np.nan), 'n_samples': len(A),
                        'n_samples_per_bin_min': min(hour_counts) if hour_counts else np.nan,
                        'low_n': len(A) < 200, 'low_n_hours': json.dumps(low_n_hours)
                    }
                    self.results['cte'].append(cte_result)
                    
                    # Save checkpoint after each CTE completion
                    self.save_checkpoint('cte', cte_result)
                    gc.collect()
                except Exception as e:
                    self.errors.append({'user_id': user_id, 'feature_mode': feature_mode, 'method': 'CTE', 'tau': tau, 'error': str(e)[:200]})
            
            # STE
            for tau in self.config['taus']:
                try:
                    logger.warning(f"STE tau={tau} START {user_id}/{feature_mode}")
                    ste = symbolic_te.run_symbolic_te_analysis(A, S, k, k, tau=tau)
                    logger.warning(f"STE tau={tau} DONE {user_id}/{feature_mode}")
                    
                    ste_result = {
                        'user_id': user_id, 'feature_mode': feature_mode, 'k': k, 'tau': tau,
                        'STE_A2S': ste.get('STE(A->S)'), 'STE_S2A': ste.get('STE(S->A)'),
                        'Delta_STE': ste.get('Delta_STE'), 'p_A2S': ste.get('p_ste(A->S)'),
                        'p_S2A': ste.get('p_ste(S->A)', np.nan)
                    }
                    self.results['ste'].append(ste_result)
                    
                    # Save checkpoint after each STE completion
                    self.save_checkpoint('ste', ste_result)
                    gc.collect()
                except Exception as e:
                    self.errors.append({'user_id': user_id, 'feature_mode': feature_mode, 'method': 'STE', 'tau': tau, 'error': str(e)[:200]})
            
            # GC
            try:
                logger.warning(f"GC START {user_id}/{feature_mode}")
                gc_res = granger_analysis.run_granger_causality(A, S, max_lag=8)
                logger.warning(f"GC DONE {user_id}/{feature_mode}")
                p_A2S = gc_res.get('gc_A_to_S_pval', np.nan)
                p_S2A = gc_res.get('gc_S_to_A_pval', np.nan)
                
                gc_result = {
                    'user_id': user_id, 'feature_mode': feature_mode,
                    'gc_optimal_lag': int(gc_res.get('gc_optimal_lag', 0)) if np.isfinite(gc_res.get('gc_optimal_lag', np.nan)) else 0,
                    'GC_A2S_pval': p_A2S, 'GC_S2A_pval': p_S2A,
                    'sign_GC': 'A2S' if (np.isfinite(p_A2S) and np.isfinite(p_S2A) and p_A2S < p_S2A) else 'S2A'
                }
                self.results['gc'].append(gc_result)
                
                # Save checkpoint after GC completion
                self.save_checkpoint('gc', gc_result)
                gc.collect()
            except Exception as e:
                self.errors.append({'user_id': user_id, 'feature_mode': feature_mode, 'method': 'GC', 'error': str(e)[:200]})
        
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            logger.warning(f"COMPLETED {user_id}/{feature_mode} at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {elapsed:.1f}s / {elapsed/60:.1f}min)")
            
            # Mark combination as completed
            self.completed_combinations.add((user_id, feature_mode))
        
        except Exception as e:
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            self.errors.append({'user_id': user_id, 'feature_mode': feature_mode, 'method': 'PROCESS', 'error': str(e)[:200]})
            logger.error(f"FAILED {user_id}/{feature_mode} at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {elapsed:.1f}s): {e}")
    
    def run(self, user_list, feature_modes):
        """Run pipeline for all users and feature modes."""
        self.start_time = datetime.now()
        self.total_users = len(user_list) * len(feature_modes)
        
        # Load checkpoint if resuming
        if self.is_resume:
            completed_count = self.load_checkpoint()
            logger.warning(f"RESUME: Skipping {completed_count} already completed combinations")
        
        # Write initial run_info
        self.write_run_info(seed=42)
        
        # Start JVM with config parameters
        jvm_cfg = self.config.get('jvm', {})
        analysis.start_jvm(
            xms=jvm_cfg.get('xms', '8g'),
            xmx=jvm_cfg.get('xmx', '16g'),
            gc_opts=jvm_cfg.get('opts', None)
        )
        logger.warning("PRODUCTION PIPELINE: Schema v1.0 with tracking + checkpointing")
        
        # Process all combinations
        for user_id in tqdm(user_list, desc='Users'):
            for feat_mode in feature_modes:
                self.process_user(user_id, feat_mode)
                self.users_completed += 1
                self.update_heartbeat(current_user=f"{user_id}/{feat_mode}")
        
        # Apply FDR and save
        self.finalize()
        
        analysis.shutdown_jvm()
    
    def finalize(self):
        """Apply FDR, save all outputs, generate final report."""
        # Convert to DataFrames
        df_te = pd.DataFrame(self.results['te'])
        df_cte = pd.DataFrame(self.results['cte'])
        df_ste = pd.DataFrame(self.results['ste'])
        df_gc = pd.DataFrame(self.results['gc'])
        df_k = pd.DataFrame(self.results['k_selected'])
        df_hbin = pd.DataFrame(self.results['hbin_counts'])
        
        # Apply FDR
        alpha = self.config['fdr']['alpha']
        if len(df_te) > 0:
            df_te = apply_fdr_per_family_tau(df_te, ['p_A2S', 'p_S2A'], ['q_A2S', 'q_S2A'], 'TE', tau_values=self.config['taus'], alpha=alpha)
            _, p_delta = compute_delta_pvalue(df_te, 'Delta_TE', 'p_Delta_TE')
            df_te['p_Delta_TE'] = p_delta
            df_te['q_Delta_TE'] = p_delta
        
        if len(df_cte) > 0:
            df_cte = apply_fdr_per_family_tau(df_cte, ['p_A2S', 'p_S2A'], ['q_A2S', 'q_S2A'], 'CTE', tau_values=self.config['taus'], alpha=alpha)
            _, p_delta = compute_delta_pvalue(df_cte, 'Delta_CTE', 'p_Delta_CTE')
            df_cte['p_Delta_CTE'] = p_delta
            df_cte['q_Delta_CTE'] = p_delta
        
        if len(df_ste) > 0:
            df_ste = apply_fdr_per_family_tau(df_ste, ['p_A2S', 'p_S2A'], ['q_A2S', 'q_S2A'], 'STE', tau_values=self.config['taus'], alpha=alpha)
            _, p_delta = compute_delta_pvalue(df_ste, 'Delta_STE', 'p_Delta_STE')
            df_ste['p_Delta_STE'] = p_delta
            df_ste['q_Delta_STE'] = p_delta
        
        if len(df_gc) > 0:
            df_gc = apply_fdr_per_family_tau(df_gc, ['GC_A2S_pval', 'GC_S2A_pval'], ['q_GC_A2S', 'q_GC_S2A'], 'GC', tau_col=None, alpha=alpha)
        
        # Save with exact schema order
        te_cols = ['user_id','feature_mode','k','l','tau','TE_A2S','TE_S2A','Delta_TE','p_A2S','p_S2A','q_A2S','q_S2A','p_Delta_TE','q_Delta_TE','n_samples','low_n']
        cte_cols = ['user_id','feature_mode','k','l','tau','hour_bins','CTE_A2S','CTE_S2A','Delta_CTE','p_A2S','p_S2A','q_A2S','q_S2A','p_Delta_CTE','q_Delta_CTE','n_samples','n_samples_per_bin_min','low_n','low_n_hours']
        ste_cols = ['user_id','feature_mode','k','tau','STE_A2S','STE_S2A','Delta_STE','p_A2S','p_S2A','q_A2S','q_S2A','p_Delta_STE','q_Delta_STE']
        gc_cols = ['user_id','feature_mode','gc_optimal_lag','GC_A2S_pval','GC_S2A_pval','q_GC_A2S','q_GC_S2A','sign_GC']
        
        df_te[te_cols].to_csv(self.out_dir / 'per_user_te.csv', index=False)
        df_cte[cte_cols].to_csv(self.out_dir / 'per_user_cte.csv', index=False)
        df_ste[ste_cols].to_csv(self.out_dir / 'per_user_ste.csv', index=False)
        df_gc[gc_cols].to_csv(self.out_dir / 'per_user_gc.csv', index=False)
        
        if len(df_k) > 0:
            df_k.to_csv(self.out_dir / 'k_selected_by_user.csv', index=False)
        
        if len(df_hbin) > 0:
            df_hbin.to_csv(self.out_dir / 'hbin_counts.csv', index=False)
        
        if self.errors:
            pd.DataFrame(self.errors).to_csv(self.out_dir / 'error_log.csv', index=False)
        
        # Final heartbeat
        self.update_heartbeat()
        
        # Update run_info with completion time
        with open(self.out_dir / 'run_info.yaml') as f:
            run_info = yaml.safe_load(f)
        run_info['completed_at'] = datetime.now().isoformat()
        run_info['duration_seconds'] = int((datetime.now() - self.start_time).total_seconds())
        run_info['users_processed'] = len(set([r['user_id'] for r in self.results['te']]))
        run_info['errors_count'] = len(self.errors)
        
        with open(self.out_dir / 'run_info.yaml', 'w') as f:
            yaml.dump(run_info, f, default_flow_style=False, sort_keys=False)
        
        logger.warning(f"Pipeline completed: {run_info['users_processed']} users, {len(self.errors)} errors")
        
        return str(self.out_dir.resolve())


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Production ExtraSensory analysis pipeline with preset configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preset configurations:
  smoke      Fast validation (2 users, k=4 fixed, 100 surrogates)
  k6_full    Pure AIS k=6 analysis (60 users, no constraints, 1000 surrogates)
  k4_fast    GUARDED_AIS k≤4 analysis (60 users, 4 modes, fast)
  24bin_cte  24-bin CTE high resolution (60 users, k≤4)

Examples:
  python run_production.py smoke
  python run_production.py k6_full --shard 0/4
  python run_production.py --config custom.yaml
        """
    )
    parser.add_argument('preset', nargs='?', help="Preset name (smoke, k6_full, k4_fast, 24bin_cte)")
    parser.add_argument('--config', help="Custom config file (overrides preset)")
    parser.add_argument('--resume', type=str, metavar='DIR', help="Resume from existing output directory")
    parser.add_argument('--shard', type=str, metavar='ID/TOTAL', help="Process shard: e.g., 0/4 means shard 0 of 4")
    parser.add_argument('--n-users', type=int, help="Override number of users from config")
    args = parser.parse_args()
    
    # Determine config file
    if args.config:
        config_path = args.config
        logger.warning(f"CUSTOM CONFIG: {config_path}")
    elif args.preset:
        preset_map = {
            'smoke': 'config/presets/smoke.yaml',
            'k6_full': 'config/presets/k6_full.yaml',
            'k4_fast': 'config/presets/k4_fast.yaml',
            '24bin_cte': 'config/presets/24bin_cte.yaml'
        }
        if args.preset not in preset_map:
            logger.error(f"Unknown preset '{args.preset}'. Available: {list(preset_map.keys())}")
            sys.exit(1)
        config_path = preset_map[args.preset]
        logger.warning(f"PRESET: {args.preset} → {config_path}")
    else:
        logger.error("No configuration specified. Use: python run_production.py <preset> or --config <file>")
        parser.print_help()
        sys.exit(1)
    
    pipeline = ProductionPipeline(config_path, resume_dir=args.resume)
    
    # Get user list
    data_root = Path(pipeline.config['data_root'])
    files = glob.glob(str(data_root / '*.features_labels.csv'))
    all_uuids = sorted([Path(f).stem.replace('.features_labels', '') for f in files])
    
    # Get n_users from config or override
    n_users = args.n_users if args.n_users else pipeline.config.get('n_users', len(all_uuids))
    user_list = all_uuids[:n_users]
    feature_modes = pipeline.config['feature_modes']
    
    logger.warning(f"CONFIG: {n_users} users, {len(feature_modes)} modes, k_strategy={pipeline.config['k_selection']['strategy']}")
    
    # Apply user sharding if specified
    if args.shard:
        try:
            shard_id, total_shards = map(int, args.shard.split('/'))
            if shard_id < 0 or shard_id >= total_shards:
                raise ValueError(f"Invalid shard_id={shard_id}, must be 0 <= shard_id < {total_shards}")
            
            # Partition users: take every Nth user starting from shard_id
            original_count = len(user_list)
            user_list = user_list[shard_id::total_shards]
            logger.warning(f"SHARD MODE: Processing shard {shard_id}/{total_shards} ({len(user_list)}/{original_count} users)")
        except Exception as e:
            logger.error(f"Failed to parse --shard argument '{args.shard}': {e}")
            logger.error("Expected format: --shard ID/TOTAL (e.g., --shard 0/4)")
            sys.exit(1)
    
    out_dir = pipeline.run(user_list, feature_modes)
    
    print(json.dumps({
        'status': 'completed',
        'OUT_DIR': out_dir,
        'config': config_path,
        'preset': args.preset if args.preset else 'custom',
        'users': len(user_list),
        'modes': len(feature_modes),
        'next_step': f'python tools/validate_outputs.py --dir {out_dir}'
    }, separators=(',', ':')))


if __name__ == "__main__":
    main()
