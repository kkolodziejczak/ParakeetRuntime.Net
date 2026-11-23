<#
.SYNOPSIS
Runs the corpus benchmark for ParakeetRuntime and NeMo-Speech.cpp on every device. See docs/benchmarks-howto.md for every option.
#>
param(
    [string] $Nemo,
    [string] $Corpus = "$env:USERPROFILE\.councils\local-parakeet-engine\fable\corpus",
    [string] $Out = (Join-Path $env:TEMP "parakeet-matrix"),
    [string] $Baseline,
    [string[]] $Languages = @("pl_pl", "en_us"),
    [string[]] $Lengths = @("wav16", "long60", "long120"),
    [string[]] $LocalDevices = @("gpu:RTX", "gpu:Intel", "cpu"),
    [string[]] $NemoDevices = @("vulkan:0", "vulkan:1"),
    [int] $Passes = 3,
    [int] $Limit = 0,
    [string] $Affinity = "C03"
)
$ErrorActionPreference = "Stop"
$project = Join-Path $PSScriptRoot "..\src\ParakeetRuntime.Cli"
$cli = Join-Path $project "bin\Release\net10.0\parakeet-runtime.exe"
dotnet build -c Release $project | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Build failed." }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$results = Join-Path $Out "matrix.tsv"

function Invoke-Bench([string] $name, [string[]] $arguments, [string] $compareWith) {
    Write-Host "=== $name"
    if ($compareWith -and (Test-Path $compareWith)) { $arguments += @("--compare", $compareWith) }
    & $cli @arguments --affinity $Affinity --results $results --out (Join-Path $Out "$name.tsv") *> (Join-Path $Out "$name.log")
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
        foreach ($device in $LocalDevices) { $runs["parakeet-$($device -replace ':', '')"] = @("--device", $device) }

        $nemoBase = Join-Path $Out "$language-$length-nemo-cpu.tsv"
        foreach ($run in $runs.Keys) {
            $name = "$language-$length-$run"
            $compareWith = if ($Baseline) { Join-Path $Baseline "$name.tsv" } elseif ($run -ne "nemo-cpu") { $nemoBase }
            if ($Baseline -and -not (Test-Path $compareWith)) { throw "The baseline has no $name.tsv; run it with the same options as this run." }
            Invoke-Bench $name ($common + $runs[$run]) $compareWith
        }
    }
}
Write-Host "Results: $results"
