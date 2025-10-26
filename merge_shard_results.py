#!/usr/bin/env python
"""Merge results from multiple shards into single output directory."""

import argparse
import pandas as pd
from pathlib import Path
import shutil
import yaml
import json

def merge_csv_files(shard_dirs, output_file, filename):
    """Merge CSV files from all shards."""
    dfs = []
    for shard_dir in shard_dirs:
        csv_path = shard_dir / filename
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"  {csv_path.name}: {len(df)} rows")
    
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_file, index=False)
        print(f"✓ Merged {filename}: {len(merged)} total rows")
        return len(merged)
    return 0

def main():
    parser = argparse.ArgumentParser(description="Merge shard results")
    parser.add_argument('--shards', nargs='+', required=True, help="Shard output directories")
    parser.add_argument('--output', required=True, help="Merged output directory")
    args = parser.parse_args()
    
    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    shard_dirs = [Path(d) for d in args.shards]
    
    print(f"Merging {len(shard_dirs)} shards into {out_dir}")
    print("")
    
    # Merge all CSV files
    csv_files = [
        'per_user_te.csv',
        'per_user_cte.csv',
        'per_user_ste.csv',
        'per_user_gc.csv',
        'k_selected_by_user.csv',
        'hbin_counts.csv',
        'error_log.csv'
    ]
    
    for csv_file in csv_files:
        print(f"Merging {csv_file}:")
        merge_csv_files(shard_dirs, out_dir / csv_file, csv_file)
        print("")
    
    # Copy run_info.yaml from first shard
    if shard_dirs[0].exists():
        run_info_src = shard_dirs[0] / 'run_info.yaml'
        if run_info_src.exists():
            shutil.copy(run_info_src, out_dir / 'run_info.yaml')
            
            # Update with shard count
            with open(out_dir / 'run_info.yaml') as f:
                run_info = yaml.safe_load(f)
            run_info['shards'] = len(shard_dirs)
            run_info['shard_dirs'] = [str(d) for d in shard_dirs]
            
            with open(out_dir / 'run_info.yaml', 'w') as f:
                yaml.dump(run_info, f, default_flow_style=False, sort_keys=False)
            
            print(f"✓ Copied run_info.yaml with shard metadata")
    
    # Merge status.json (show aggregate progress)
    statuses = []
    for shard_dir in shard_dirs:
        status_path = shard_dir / 'status.json'
        if status_path.exists():
            with open(status_path) as f:
                statuses.append(json.load(f))
    
    if statuses:
        total_done = sum(s['progress']['done'] for s in statuses)
        total_users = sum(s['progress']['total'] for s in statuses)
        total_errors = sum(s['errors_count'] for s in statuses)
        
        merged_status = {
            'status': 'merged',
            'shards': len(statuses),
            'progress': {
                'done': total_done,
                'total': total_users,
                'percent': round(100 * total_done / total_users, 1) if total_users > 0 else 0
            },
            'errors_count': total_errors
        }
        
        with open(out_dir / 'status.json', 'w') as f:
            json.dump(merged_status, f, indent=2)
        
        print(f"✓ Merged status.json: {total_done}/{total_users} users ({merged_status['progress']['percent']}%)")
    
    print("")
    print(f"Merge complete! Results in: {out_dir.resolve()}")
    print(f"Next: python tools/validate_outputs.py --dir {out_dir}")

if __name__ == "__main__":
    main()
