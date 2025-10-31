#!/usr/bin/env python
"""Schema-compliant SMOKE test: 2 users, composite only, contract v1.0.

Generates outputs matching the exact schema specification for validation.
"""
import sys, json, logging, glob, gc, yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))

from src import preprocessing, analysis, granger_analysis, symbolic_te
from src.fdr_utils import apply_fdr_per_family_tau, compute_delta_pvalue

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Load config
with open('config/proposal.yaml') as f:
    CONFIG = yaml.safe_load(f)

def main():
    # Setup
    out_dir = Path(f'analysis/out/smoke_corrected_{datetime.now().strftime("%Y%m%d_%H%M")}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Start JVM
    analysis.start_jvm()
    logger.warning("SMOKE CORRECTED: Schema v1.0 compliance test")
    
    # Get 2 users
    data_root = Path(CONFIG['data_root'])
    files = glob.glob(str(data_root / '*.features_labels.csv'))
    uuids = sorted([Path(f).stem.replace('.features_labels', '') for f in files])[:CONFIG['smoke']['n_users']]
    
    results = {'te': [], 'cte': [], 'ste': [], 'gc': [], 'errors': []}
    k_fixed = CONFIG['smoke']['k_fixed']
    
    for uuid in uuids:
        for feat_mode in CONFIG['smoke']['feature_modes']:
            try:
                # Load data
                raw = preprocessing.load_subject_data(uuid)
                A, S, H_raw, _ = preprocessing.create_variables(raw, feature_mode=feat_mode)
                
                if len(A) == 0:
                    results['errors'].append({'user_id': uuid, 'error': 'empty_data'})
                    continue
                
                base_A, base_S = int(np.max(A)) + 1, int(np.max(S)) + 1
                k = l = k_fixed
                
                # TE with native tau
                for tau in CONFIG['taus']:
                    try:
                        te = analysis.run_te_analysis(A.astype(int), S.astype(int), k, l, base_A, base_S, tau=tau)
                        results['te'].append({
                            'user_id': uuid,
                            'feature_mode': feat_mode,
                            'k': k,
                            'l': l,
                            'tau': tau,
                            'TE_A2S': te.get('TE(A->S)'),
                            'TE_S2A': te.get('TE(S->A)'),
                            'Delta_TE': te.get('Delta_TE'),
                            'p_A2S': te.get('p(A->S)'),
                            'p_S2A': te.get('p(S->A)', np.nan),  # Need to add to analysis.py
                            'n_samples': len(A),
                            'low_n': len(A) < 100
                        })
                        gc.collect()
                    except Exception as e:
                        results['errors'].append({'user_id': uuid, 'method': 'TE', 'tau': tau, 'error': str(e)[:100]})
                
                # CTE with data-level lag
                for tau in CONFIG['taus']:
                    try:
                        cte = analysis.run_cte_analysis(A.astype(int), S.astype(int), H_raw, k, l, base_A, base_S, CONFIG['hour_bins'], tau=tau)
                        results['cte'].append({
                            'user_id': uuid,
                            'feature_mode': feat_mode,
                            'k': k,
                            'l': l,
                            'tau': tau,
                            'hour_bins': CONFIG['hour_bins'],
                            'CTE_A2S': cte.get('CTE(A->S|H_bin)'),
                            'CTE_S2A': cte.get('CTE(S->A|H_bin)'),
                            'Delta_CTE': cte.get('Delta_CTE_bin'),
                            'p_A2S': cte.get('p_cte(A->S|H_bin)'),
                            'p_S2A': cte.get('p_cte(S->A|H_bin)', np.nan),
                            'n_samples': len(A),
                            'n_samples_per_bin_min': cte.get('n_samples_per_bin_min', np.nan),
                            'low_n': len(A) < 200,
                            'low_n_hours': cte.get('low_n_hours', [])
                        })
                        gc.collect()
                    except Exception as e:
                        results['errors'].append({'user_id': uuid, 'method': 'CTE', 'tau': tau, 'error': str(e)[:100]})
                
                # STE with native tau
                for tau in CONFIG['taus']:
                    try:
                        ste = symbolic_te.run_symbolic_te_analysis(A, S, k, k, tau=tau)
                        results['ste'].append({
                            'user_id': uuid,
                            'feature_mode': feat_mode,
                            'k': k,
                            'tau': tau,
                            'STE_A2S': ste.get('STE(A->S)'),
                            'STE_S2A': ste.get('STE(S->A)'),
                            'Delta_STE': ste.get('Delta_STE'),
                            'p_A2S': ste.get('p_ste(A->S)'),
                            'p_S2A': ste.get('p_ste(S->A)', np.nan)
                        })
                        gc.collect()
                    except Exception as e:
                        results['errors'].append({'user_id': uuid, 'method': 'STE', 'tau': tau, 'error': str(e)[:100]})
                
                # GC (no tau)
                try:
                    gc_res = granger_analysis.run_granger_causality(A, S, max_lag=8)
                    p_A2S = gc_res.get('gc_A_to_S_pval', np.nan)
                    p_S2A = gc_res.get('gc_S_to_A_pval', np.nan)
                    results['gc'].append({
                        'user_id': uuid,
                        'feature_mode': feat_mode,
                        'gc_optimal_lag': int(gc_res.get('gc_optimal_lag', 0)) if np.isfinite(gc_res.get('gc_optimal_lag', np.nan)) else 0,
                        'GC_A2S_pval': p_A2S,
                        'GC_S2A_pval': p_S2A,
                        'sign_GC': 'A2S' if (np.isfinite(p_A2S) and np.isfinite(p_S2A) and p_A2S < p_S2A) else 'S2A'
                    })
                    gc.collect()
                except Exception as e:
                    results['errors'].append({'user_id': uuid, 'method': 'GC', 'error': str(e)[:100]})
            
            except Exception as e:
                results['errors'].append({'user_id': uuid, 'feature_mode': feat_mode, 'error': str(e)[:100]})
    
    # Convert to DataFrames
    df_te = pd.DataFrame(results['te'])
    df_cte = pd.DataFrame(results['cte'])
    df_ste = pd.DataFrame(results['ste'])
    df_gc = pd.DataFrame(results['gc'])
    
    # Apply per-(family, tau) FDR
    if len(df_te) > 0:
        df_te = apply_fdr_per_family_tau(
            df_te,
            p_cols=['p_A2S', 'p_S2A'],
            q_cols=['q_A2S', 'q_S2A'],
            family='TE',
            tau_values=CONFIG['taus'],
            alpha=CONFIG['fdr']['alpha']
        )
        # Group-level Delta p-value
        median_delta, p_delta = compute_delta_pvalue(df_te, 'Delta_TE', 'p_Delta_TE')
        df_te['p_Delta_TE'] = p_delta
        df_te['q_Delta_TE'] = p_delta  # Single value, no FDR needed
    
    if len(df_cte) > 0:
        df_cte = apply_fdr_per_family_tau(
            df_cte,
            p_cols=['p_A2S', 'p_S2A'],
            q_cols=['q_A2S', 'q_S2A'],
            family='CTE',
            tau_values=CONFIG['taus'],
            alpha=CONFIG['fdr']['alpha']
        )
        median_delta, p_delta = compute_delta_pvalue(df_cte, 'Delta_CTE', 'p_Delta_CTE')
        df_cte['p_Delta_CTE'] = p_delta
        df_cte['q_Delta_CTE'] = p_delta
    
    if len(df_ste) > 0:
        df_ste = apply_fdr_per_family_tau(
            df_ste,
            p_cols=['p_A2S', 'p_S2A'],
            q_cols=['q_A2S', 'q_S2A'],
            family='STE',
            tau_values=CONFIG['taus'],
            alpha=CONFIG['fdr']['alpha']
        )
        median_delta, p_delta = compute_delta_pvalue(df_ste, 'Delta_STE', 'p_Delta_STE')
        df_ste['p_Delta_STE'] = p_delta
        df_ste['q_Delta_STE'] = p_delta
    
    if len(df_gc) > 0:
        df_gc = apply_fdr_per_family_tau(
            df_gc,
            p_cols=['GC_A2S_pval', 'GC_S2A_pval'],
            q_cols=['q_GC_A2S', 'q_GC_S2A'],
            family='GC',
            tau_col=None,  # No tau for GC
            alpha=CONFIG['fdr']['alpha']
        )
    
    # Save with exact schema order
    te_cols = ['user_id', 'feature_mode', 'k', 'l', 'tau', 'TE_A2S', 'TE_S2A', 'Delta_TE',
               'p_A2S', 'p_S2A', 'q_A2S', 'q_S2A', 'p_Delta_TE', 'q_Delta_TE', 'n_samples', 'low_n']
    cte_cols = ['user_id', 'feature_mode', 'k', 'l', 'tau', 'hour_bins', 'CTE_A2S', 'CTE_S2A', 'Delta_CTE',
                'p_A2S', 'p_S2A', 'q_A2S', 'q_S2A', 'p_Delta_CTE', 'q_Delta_CTE',
                'n_samples', 'n_samples_per_bin_min', 'low_n', 'low_n_hours']
    ste_cols = ['user_id', 'feature_mode', 'k', 'tau', 'STE_A2S', 'STE_S2A', 'Delta_STE',
                'p_A2S', 'p_S2A', 'q_A2S', 'q_S2A', 'p_Delta_STE', 'q_Delta_STE']
    gc_cols = ['user_id', 'feature_mode', 'gc_optimal_lag', 'GC_A2S_pval', 'GC_S2A_pval',
               'q_GC_A2S', 'q_GC_S2A', 'sign_GC']
    
    df_te[te_cols].to_csv(out_dir / 'per_user_te.csv', index=False)
    df_cte[cte_cols].to_csv(out_dir / 'per_user_cte.csv', index=False)
    df_ste[ste_cols].to_csv(out_dir / 'per_user_ste.csv', index=False)
    df_gc[gc_cols].to_csv(out_dir / 'per_user_gc.csv', index=False)
    
    if results['errors']:
        pd.DataFrame(results['errors']).to_csv(out_dir / 'error_log.csv', index=False)
    
    # Save config
    with open(out_dir / 'config_used.yaml', 'w') as f:
        yaml.dump(CONFIG, f)
    
    analysis.shutdown_jvm()
    
    # Report
    report = {
        'status': 'done',
        'phase': 'SMOKE_CORRECTED',
        'schema_version': CONFIG['schema_version'],
        'OUT_DIR': str(out_dir.resolve()),
        'coverage': {'users': CONFIG['smoke']['n_users'], 'modes': len(CONFIG['smoke']['feature_modes'])},
        'params': {'k_fixed': k_fixed, 'taus': CONFIG['taus'], 'hour_bins': CONFIG['hour_bins']},
        'results_summary': {
            'TE': {'median_Delta': float(df_te['Delta_TE'].median()) if len(df_te) > 0 else np.nan, 'n': len(df_te)},
            'CTE': {'median_Delta': float(df_cte['Delta_CTE'].median()) if len(df_cte) > 0 else np.nan, 'n': len(df_cte)},
            'STE': {'median_Delta': float(df_ste['Delta_STE'].median()) if len(df_ste) > 0 else np.nan, 'n': len(df_ste)},
            'GC': {'n': len(df_gc)}
        },
        'errors': len(results['errors']),
        'next_step': f'python tools/validate_outputs.py --dir {out_dir}'
    }
    
    print(json.dumps(report, separators=(',', ':')))
    return report

if __name__ == "__main__":
    main()
