<#
.SYNOPSIS
  Monitor shard output directories and log progress periodically.

.DESCRIPTION
  Scans shard directories matching a pattern, reads status.json where available,
  and appends a one-line progress snapshot to a UTF-8 log file on each interval.
  Exits when all shards report completion or continue until manually stopped.

.PARAMETER Pattern
  Glob pattern to find shard directories (default: analysis/out/*_shard*).

.PARAMETER IntervalSeconds
  Interval in seconds between scans (default: 1800).

.PARAMETER LogPath
  Path to the log file to append (default: monitor.log).

.NOTES
  Archived-style helper script; adjust to your environment as needed.
#>

param(
  [string]$Pattern = "analysis/out/*_shard*",
  [int]$IntervalSeconds = 1800,
  [string]$LogPath = "monitor.log"
)

Write-Host "Monitoring shards with pattern '$Pattern' every $IntervalSeconds s. Log: $LogPath"

while ($true) {
  $dirs = Get-ChildItem $Pattern -Directory -ErrorAction SilentlyContinue
  $timestamp = Get-Date -Format s
  $lines = @()
  $allDone = $false
  if ($dirs -and $dirs.Count -gt 0) {
    $allDone = $true
    foreach ($d in $dirs) {
      $status = Join-Path $d.FullName 'status.json'
      if (Test-Path $status) {
        try { $j = Get-Content $status -Raw | ConvertFrom-Json } catch { $j = $null }
        if ($j) {
          $total = $j.pipeline.total_users
          $comp = $j.pipeline.completed_users
          $pct = $j.pipeline.progress_percent
          $stage = if ($j.current_task) { $j.current_task.stage } else { '' }
          $user  = if ($j.current_task) { $j.current_task.user_id } else { '' }
          if (-not ($comp -ge $total)) { $allDone = $false }
          $lines += "${timestamp} $($d.Name) $comp/$total (${pct}%) stage='$stage' user='$user'"
        } else {
          $allDone = $false
          $lines += "${timestamp} $($d.Name) status=unreadable"
        }
      } else {
        $allDone = $false
        $lines += "${timestamp} $($d.Name) status.json missing"
      }
    }
  } else {
    $lines += "${timestamp} no matching shard directories"
  }

  $lines | Out-File -Append -FilePath $LogPath -Encoding utf8
  if ($allDone -and $dirs -and $dirs.Count -gt 0) { break }
  Start-Sleep -Seconds $IntervalSeconds
}

Write-Host "Monitoring complete."
