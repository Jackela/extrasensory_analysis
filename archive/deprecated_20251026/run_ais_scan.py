#!/usr/bin/env python
"""AIS-only scan for all 60 users to determine optimal k distribution.

Runs AIS k-selection for k∈{1,2,3,4,5,6} with 100 surrogates.
Outputs k_selected_by_user.csv and JSON summary without computing TE/CTE/STE/GC.
"""
import sys, json, glob
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))

from src import preprocessing, analysis
from src.k_selection import select_k_via_ais

def main():
    # Setup
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_dir = Path(f'analysis/out/ais_scan_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all users
    data_root = Path('data/ExtraSensory.per_uuid_features_labels')
    files = glob.glob(str(data_root / '*.features_labels.csv'))
    all_uuids = sorted([Path(f).stem.replace('.features_labels', '') for f in files])[:60]
    
    # Start JVM
    analysis.start_jvm(xms="8g", xmx="16g")
    
    # AIS scan
    results = []
    k_grid = [1, 2, 3, 4, 5, 6]
    epsilon_bits = 0.005
    
    for user_id in tqdm(all_uuids, desc='AIS scan'):
        try:
            # Load data (use composite feature mode)
            raw = preprocessing.load_subject_data(user_id)
            A, S, H_raw, _ = preprocessing.create_variables(raw, feature_mode='composite')
            
            if len(S) == 0:
                continue
            
            base_S = int(np.max(S)) + 1
            
            # Run AIS k-selection
            k_info = select_k_via_ais(
                S.astype(int), 
                base_S, 
                k_grid,
                num_surrogates=100,
                criterion='max_ais'
            )
            
            results.append({
                'user_id': user_id,
                'k_selected': k_info['k_selected'],
                'k_grid': json.dumps(k_grid),
                'ais_curve': json.dumps(k_info['ais_values'])
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
    k6_users = df[df['k_selected'] == 6]['user_id'].tolist()
    
    summary = {
        'total_users': len(df),
        'k_distribution': k_counts,
        'k6_users': k6_users,
        'k6_count': len(k6_users),
        'output_dir': str(out_dir.resolve())
    }
    
    print(json.dumps(summary, separators=(',', ':')))

if __name__ == "__main__":
    main()
