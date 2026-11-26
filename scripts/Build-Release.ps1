# Builds the unsigned release files for one version into artifacts\release\<version>:
#
#   ParakeetRuntime.Net-<version>-win-x64\   parakeet-runtime.exe (self-contained, single file),
#                                        README.md, LICENSE, THIRD-PARTY-NOTICES.md
#   ParakeetRuntime.Net.<version>.nupkg      the library package
#
# The release workflow signs these, then zips the folder. Test-ReleaseArtifacts.ps1 checks them.
param(
    [Parameter(Mandatory = $true)] [ValidatePattern('^\d+\.\d+\.\d+$')] [string]$Version
)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$output = Join-Path $root "artifacts\release\$Version"
# The exe is compressed inside (100 to 48 MB on disk); see docs/benchmarks.md, 2026-11-26.
$folder = Join-Path $output "ParakeetRuntime.Net-$Version-win-x64"
if (Test-Path -LiteralPath $output) { Remove-Item -LiteralPath $output -Recurse -Force }

dotnet publish (Join-Path $root 'src\ParakeetRuntime.Net.Cli\ParakeetRuntime.Net.Cli.csproj') --configuration Release --runtime win-x64 --self-contained -p:PublishSingleFile=true -p:EnableCompressionInSingleFile=true -p:Version=$Version --output $folder
if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed with exit code $LASTEXITCODE." }

dotnet pack (Join-Path $root 'src\ParakeetRuntime.Net\ParakeetRuntime.Net.csproj') --configuration Release -p:Version=$Version --output $output
if ($LASTEXITCODE -ne 0) { throw "dotnet pack failed with exit code $LASTEXITCODE." }

foreach ($document in 'README.md', 'LICENSE', 'THIRD-PARTY-NOTICES.md') {
    Copy-Item -LiteralPath (Join-Path $root $document) -Destination $folder
}
Write-Host "Unsigned release files: $output"
