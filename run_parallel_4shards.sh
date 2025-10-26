#!/usr/bin/env bash
# Launch 4 parallel processes for k=6 analysis
# Usage: ./run_parallel_4shards.sh [preset]
# Default preset: k6_full

set -euo pipefail

PRESET="${1:-k6_full}"

echo "Starting 4-process parallel execution with preset: $PRESET"
echo "Each process will use config from: config/presets/${PRESET}.yaml"
echo ""

# Create analysis directory if not exists
mkdir -p analysis

# Launch 4 parallel processes
python run_production.py "$PRESET" --shard 0/4 > analysis/shard0.log 2>&1 &
SHARD0_PID=$!
echo "Shard 0/4 started (PID: $SHARD0_PID)"

python run_production.py "$PRESET" --shard 1/4 > analysis/shard1.log 2>&1 &
SHARD1_PID=$!
echo "Shard 1/4 started (PID: $SHARD1_PID)"

python run_production.py "$PRESET" --shard 2/4 > analysis/shard2.log 2>&1 &
SHARD2_PID=$!
echo "Shard 2/4 started (PID: $SHARD2_PID)"

python run_production.py "$PRESET" --shard 3/4 > analysis/shard3.log 2>&1 &
SHARD3_PID=$!
echo "Shard 3/4 started (PID: $SHARD3_PID)"

echo ""
echo "4 processes launched with preset: $PRESET"
echo ""
echo "Monitor progress:"
echo "  tail -f analysis/shard0.log"
echo "  tail -f analysis/shard1.log"
echo "  tail -f analysis/shard2.log"
echo "  tail -f analysis/shard3.log"
echo ""
echo "Kill all shards: kill $SHARD0_PID $SHARD1_PID $SHARD2_PID $SHARD3_PID"
echo ""

# Wait for all processes to complete
wait $SHARD0_PID $SHARD1_PID $SHARD2_PID $SHARD3_PID

echo "All shards completed!"
