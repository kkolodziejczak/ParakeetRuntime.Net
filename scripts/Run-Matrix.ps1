# Runs the corpus benchmark for both engines on every device, one run at a time, pinned to 4 cores.
# Every run appends a row to <Out>\matrix.tsv; transcripts (<run>.tsv) and logs (<run>.log) go next to it.
#   Full matrix: powershell -File src/LocalBuddy.Spike/Run-Matrix.ps1 -Nemo <nemo-speech.exe> [-Out <folder>]
#     NeMo-Speech.cpp on the CPU runs first for each corpus; every other run is compared with its transcripts.
#   Quick A/B of a change: run without -Nemo on a subset, then again from the changed checkout with -Baseline:
#     ... Run-Matrix.ps1 -Languages pl_pl -Lengths long120 -LocalDevices cpu -Limit 3 -Out before
#     <changed checkout>\src\LocalBuddy.Spike\Run-Matrix.ps1 (same options) -Out after -Baseline before
#     Each run is then compared with the transcript of the same run in the baseline folder.
param(
    # Without it only LocalBuddy runs.
    [string] $Nemo,
    [string] $Corpus = "$env:USERPROFILE\.councils\local-parakeet-engine\fable\corpus",
    [string] $Out = (Join-Path $env:TEMP "localbuddy-matrix"),
    [string] $Baseline,
    [string[]] $Languages = @("pl_pl", "en_us"),
    [string[]] $Lengths = @("wav16", "long60", "long120"),
    # LocalBuddy devices (gpu:<part of the adapter name>) and the NeMo-Speech.cpp devices on the same hardware.
    [string[]] $LocalDevices = @("gpu:RTX", "gpu:Intel", "cpu"),
    [string[]] $NemoDevices = @("vulkan:0", "vulkan:1"),
    # Every run keeps its best pass; long CPU runs vary by up to 40% between passes.
    [int] $Passes = 3,
    # The first N clips of each corpus only (0: all of them).
    [int] $Limit = 0,
    # 4 P-cores of the Core Ultra 7 270K Plus (CPUs 0, 1, 10, 11). F0, used until 2026-11-22, was 4 E-cores.
    [string] $Affinity = "C03"
)
$ErrorActionPreference = "Stop"
$cli = Join-Path $PSScriptRoot "..\src\ParakeetRuntime.Cli"
$spike = Join-Path $cli "bin\Release\net10.0\parakeet-runtime.exe"
dotnet build -c Release $cli | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Build failed." }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$results = Join-Path $Out "matrix.tsv"

function Invoke-Bench([string] $name, [string[]] $arguments, [string] $compareWith) {
    Write-Host "=== $name"
    if ($compareWith -and (Test-Path $compareWith)) { $arguments += @("--compare", $compareWith) }
    & $spike @arguments --affinity $Affinity --results $results --out (Join-Path $Out "$name.tsv") *> (Join-Path $Out "$name.log")
    if ($LASTEXITCODE -ne 0) { throw "$name failed, see $Out\$name.log" }
    Get-Content (Join-Path $Out "$name.log") | Select-String "^pass|WER|identical|memory peak" | ForEach-Object { "    $_" }
}

foreach ($language in $Languages) {
    foreach ($length in $Lengths) {
        $folder = Join-Path $Corpus "$language\$length"
        $reference = if ($length -eq "wav16") { Join-Path $Corpus "$language\dev.tsv" } else { Join-Path $folder "dev.tsv" }
        $common = @("--corpus", (Join-Path $folder "list.txt"), "--reference", $reference, "--passes", $Passes)
        if ($length -eq "wav16") { $common += @("--max-seconds", "20") }
        if ($Limit -gt 0) { $common += @("--limit", $Limit) }

        $runs = [ordered]@{}
        if ($Nemo) {
            $runs["nemo-cpu"] = @("--nemo", $Nemo, "--device", "cpu")
            foreach ($device in $NemoDevices) { $runs["nemo-$($device -replace ':', '')"] = @("--nemo", $Nemo, "--device", $device) }
        }
        foreach ($device in $LocalDevices) { $runs["localbuddy-$($device -replace ':', '')"] = @("--device", $device) }

        $nemoBase = Join-Path $Out "$language-$length-nemo-cpu.tsv"
        foreach ($run in $runs.Keys) {
            $name = "$language-$length-$run"
            $compareWith = if ($Baseline) { Join-Path $Baseline "$name.tsv" } elseif ($run -ne "nemo-cpu") { $nemoBase }
            Invoke-Bench $name ($common + $runs[$run]) $compareWith
        }
    }
}
Write-Host "Results: $results"
