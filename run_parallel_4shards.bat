@echo off
REM Launch 4 parallel processes for k=6 analysis
REM Each process: 12GB heap, handles 15 users, ~60 hours total

echo Starting 4-process parallel execution...
echo Each process will use 12GB heap (48GB total)
echo Expected completion: ~60 hours

start "Shard 0/4" cmd /c "python run_production.py --full --shard 0/4 > analysis\shard0.log 2>&1"
start "Shard 1/4" cmd /c "python run_production.py --full --shard 1/4 > analysis\shard1.log 2>&1"
start "Shard 2/4" cmd /c "python run_production.py --full --shard 2/4 > analysis\shard2.log 2>&1"
start "Shard 3/4" cmd /c "python run_production.py --full --shard 3/4 > analysis\shard3.log 2>&1"

echo.
echo 4 processes launched. Monitor progress:
echo   - analysis\shard0.log
echo   - analysis\shard1.log
echo   - analysis\shard2.log
echo   - analysis\shard3.log
echo.
echo Each shard processes 15 users (60 users total / 4 processes)
echo Use Ctrl+C in each window to stop individual processes
pause
