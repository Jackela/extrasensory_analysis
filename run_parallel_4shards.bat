@echo off
REM Launch 4 parallel processes for k=6 analysis
REM Usage: run_parallel_4shards.bat [preset]
REM Default preset: k6_full

set PRESET=%1
if "%PRESET%"=="" set PRESET=k6_full

echo Starting 4-process parallel execution with preset: %PRESET%
echo Each process will use config from: config\presets\%PRESET%.yaml
echo.

if not exist analysis mkdir analysis

start "Shard 0/4" cmd /c "python run_production.py %PRESET% --shard 0/4 > analysis\shard0.log 2>&1"
start "Shard 1/4" cmd /c "python run_production.py %PRESET% --shard 1/4 > analysis\shard1.log 2>&1"
start "Shard 2/4" cmd /c "python run_production.py %PRESET% --shard 2/4 > analysis\shard2.log 2>&1"
start "Shard 3/4" cmd /c "python run_production.py %PRESET% --shard 3/4 > analysis\shard3.log 2>&1"

echo.
echo 4 processes launched with preset: %PRESET%
echo.
echo Monitor progress:
echo   - analysis\shard0.log
echo   - analysis\shard1.log
echo   - analysis\shard2.log
echo   - analysis\shard3.log
echo.
echo Use Ctrl+C in each window to stop individual processes
pause
