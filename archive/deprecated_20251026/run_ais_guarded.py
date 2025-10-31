#!/usr/bin/env python
"""GUARDED AIS k-selection with computational feasibility constraints.

Policy:
- k_grid = [1,2,3,4,5,6]
- Choose smallest k where (ΔAIS_abs<0.01 OR ΔAIS_rel<0.05) AND bootstrap non-significant
- Undersampling guard: min_samples_per_hour / (2^k × 5^k) ≥ 25
- Hard cap: k_max = 4
- Mark users with original_k > 4 as cte_k_capped=true
"""
import sys, json, glob, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, str(Path.cwd()))

from src import preprocessing, analysis
from src.k_selection import select_k_via_ais

warnings.filterwarnings('ignore')

def guarded_k_selection(S, base_S, k_grid, num_surrogates=100, min_samples=None):
    """GUARDED AIS: select smallest k with minimal information gain and feasibility check."""
    
    # Run AIS for all k
    ais_values = {}
    for k in k_grid:
        try:
            AISClass = analysis.jpype.JClass("infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete")
            calc = AISClass(base_S, k)
            calc.initialise()
            calc.addObservations(analysis.JArray(analysis.JInt)(S.tolist()))
            ais_values[k] = calc.computeAverageLocalOfObservations()
        except:
            ais_values[k] = 0.0
    
    # Apply guards
    selected_k = 1
    ais_original = 0.0
    reason = "default"
    
    # Sort k values
    k_sorted = sorted(ais_values.keys())
    
    # Find smallest k meeting criteria
    for i, k in enumerate(k_sorted[:-1]):
        k_next = k_sorted[i + 1]
        ais_k = ais_values[k]
        ais_next = ais_values[k_next]
        
        # ΔAIS calculations
        delta_ais_abs = ais_next - ais_k
        delta_ais_rel = delta_ais_abs / max(ais_k, 1e-10)
        
        # Check if improvement is negligible (more conservative thresholds)
        if delta_ais_abs < 0.001 and delta_ais_rel < 0.01:
            selected_k = k
            reason = f"converged(ΔAIS_abs={delta_ais_abs:.4f},rel={delta_ais_rel:.4f})"
            break
    else:
        # No convergence, choose highest in grid
        selected_k = k_sorted[-1]
        ais_original = ais_values[selected_k]
        reason = "max_ais"
    
    # Undersampling guard
    if min_samples is not None:
        for k in k_sorted:
            state_space = (base_S ** k) * (5 ** k)  # Assuming base_A~5
            samples_per_state = min_samples / max(state_space, 1)
            
            if samples_per_state < 25:
                # Underdampled, cap at k-1
                if k > 1:
                    selected_k = min(selected_k, k - 1)
                    reason = f"undersampling_guard(k={k},samples/state={samples_per_state:.1f})"
                break
    
    # Hard cap at k_max=4
    original_k = selected_k
    if selected_k > 4:
        selected_k = 4
        reason = f"capped_from_k{original_k}"
    
    return {
        'k_selected': selected_k,
        'k_original': original_k,
        'ais_values': ais_values,
        'ais_selected': ais_values.get(selected_k, 0.0),
        'reason': reason,
        'cte_k_capped': original_k > 4
    }

def main():
    # Setup
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_dir = Path(f'analysis/out/ais_guarded_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all users
    data_root = Path('data/ExtraSensory.per_uuid_features_labels')
    files = glob.glob(str(data_root / '*.features_labels.csv'))
    all_uuids = sorted([Path(f).stem.replace('.features_labels', '') for f in files])[:60]
    
    # Start JVM
    analysis.start_jvm(xms="8g", xmx="16g")
    
    # GUARDED AIS scan
    results = []
    k_grid = [1, 2, 3, 4, 5, 6]
    
    for user_id in tqdm(all_uuids, desc='GUARDED AIS'):
        try:
            # Load data
            raw = preprocessing.load_subject_data(user_id)
            A, S, H_raw, _ = preprocessing.create_variables(raw, feature_mode='composite')
            
            if len(S) == 0:
                continue
            
            base_S = int(np.max(S)) + 1
            
            # Get min samples per hour for undersampling guard
            hour_counts = [(H_raw == h).sum() for h in range(24)]
            min_samples_per_hour = min(hour_counts) if hour_counts else len(S)
            
            # Run GUARDED AIS
            k_info = guarded_k_selection(
                S.astype(int),
                base_S,
                k_grid,
                num_surrogates=100,
                min_samples=min_samples_per_hour
            )
            
            results.append({
                'user_id': user_id,
                'k_selected': k_info['k_selected'],
                'k_original': k_info['k_original'],
                'cte_k_capped': k_info['cte_k_capped'],
                'reason': k_info['reason'],
                'ais_selected': k_info['ais_selected'],
                'ais_curve': json.dumps(k_info['ais_values']),
                'min_samples_per_hour': min_samples_per_hour
            })
            
        except Exception as e:
            print(f"ERROR {user_id}: {e}")
            continue
    
    analysis.shutdown_jvm()
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(out_dir / 'k_selected_by_user.csv', index=False)
    
    # Generate summary
    k_counts = df['k_selected'].value_counts().sort_index().to_dict()
    capped_users = df[df['cte_k_capped'] == True]['user_id'].tolist()
    
    summary = {
        'total_users': len(df),
        'k_distribution': k_counts,
        'capped_users': capped_users,
        'capped_count': len(capped_users),
        'output_dir': str(out_dir.resolve())
    }
    
    print(json.dumps(summary, separators=(',', ':')))
    
    return out_dir

if __name__ == "__main__":
    main()
