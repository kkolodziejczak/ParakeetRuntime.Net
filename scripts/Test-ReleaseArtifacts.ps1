# Checks the release files in artifacts\release\<version> that Build-Release.ps1 made, before
# signing and again after it:
#
#   - the folder holds exactly parakeet-runtime.exe and its documents, and the exe starts and
#     reports the version;
#   - the package is ParakeetRuntime.Net <version> with the library, its readme and the notices;
#   - with -ExpectedSignerThumbprint, the exe, the DLL inside the package and the package itself
#     are signed by that certificate, with a timestamp.
param(
    [Parameter(Mandatory = $true)] [ValidatePattern('^\d+\.\d+\.\d+$')] [string]$Version,
    [string]$ExpectedSignerThumbprint = ''
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
# SignedCms ships with PowerShell 7 as System.Security.Cryptography.Pkcs, and with .NET Framework
# (Windows PowerShell 5.1) inside System.Security.
try { Add-Type -AssemblyName System.Security.Cryptography.Pkcs } catch { Add-Type -AssemblyName System.Security }

$root = Split-Path -Parent $PSScriptRoot
$output = Join-Path $root "artifacts\release\$Version"
$folder = Join-Path $output "ParakeetRuntime.Net-$Version-win-x64"
$exe = Join-Path $folder 'parakeet-runtime.exe'
$packagePath = Join-Path $output "ParakeetRuntime.Net.$Version.nupkg"

function Assert-Signed([string]$Path) {
    $signature = Get-AuthenticodeSignature -LiteralPath $Path
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Thumbprint -ne $ExpectedSignerThumbprint) {
        throw "'$Path' signature is '$($signature.Status)' from '$($signature.SignerCertificate.Subject)'; expected certificate $ExpectedSignerThumbprint."
    }
    if ($null -eq $signature.TimeStamperCertificate) { throw "'$Path' signature has no timestamp." }
    return $signature.SignerCertificate
}

$expectedFiles = @('LICENSE', 'README.md', 'THIRD-PARTY-NOTICES.md', 'parakeet-runtime.exe')
$actualFiles = @(Get-ChildItem -LiteralPath $folder -Recurse -File | ForEach-Object { $_.FullName.Substring($folder.Length + 1) })
if (Compare-Object $expectedFiles $actualFiles) {
    throw "The release folder holds '$($actualFiles -join "', '")'; expected '$($expectedFiles -join "', '")'."
}

$reported = (& $exe --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $reported -notmatch "^$([regex]::Escape($Version))(\+|$)") {
    throw "parakeet-runtime.exe --version exited with $LASTEXITCODE and printed '$reported'; expected version $Version."
}

$packages = @(Get-ChildItem -LiteralPath $output -Filter '*.nupkg' -File | ForEach-Object Name)
if ($packages.Count -ne 1 -or $packages[0] -ne "ParakeetRuntime.Net.$Version.nupkg") {
    throw "Expected only ParakeetRuntime.Net.$Version.nupkg in $output; found '$($packages -join "', '")'."
}

$unpacked = Join-Path ([IO.Path]::GetTempPath()) "parakeet-runtime-package-$([guid]::NewGuid().ToString('N'))"
try {
    [IO.Compression.ZipFile]::ExtractToDirectory($packagePath, $unpacked)
    $metadata = ([xml](Get-Content -LiteralPath (Join-Path $unpacked 'ParakeetRuntime.Net.nuspec') -Raw)).package.metadata
    if ($metadata.id -ne 'ParakeetRuntime.Net' -or $metadata.version -ne $Version) {
        throw "The package is '$($metadata.id)' $($metadata.version); expected ParakeetRuntime.Net $Version."
    }
    $library = Join-Path $unpacked 'lib\net10.0\ParakeetRuntime.Net.dll'
    foreach ($required in $library, (Join-Path $unpacked 'README.md'), (Join-Path $unpacked 'THIRD-PARTY-NOTICES.md')) {
        if (-not (Test-Path -LiteralPath $required)) { throw "The package has no $($required.Substring($unpacked.Length + 1))." }
    }

    if ($ExpectedSignerThumbprint) {
        Assert-Signed $exe | Out-Null
        $certificate = Assert-Signed $library
        $fingerprint = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($certificate.RawData)) -replace '-', ''
        dotnet nuget verify $packagePath --all --certificate-fingerprint $fingerprint
        if ($LASTEXITCODE -ne 0) { throw "dotnet nuget verify failed with exit code $LASTEXITCODE." }
        # dotnet nuget verify only warns (NU3027) when the package signature has no timestamp.
        $cms = New-Object Security.Cryptography.Pkcs.SignedCms
        $cms.Decode([IO.File]::ReadAllBytes((Join-Path $unpacked '.signature.p7s')))
        if (-not ($cms.SignerInfos[0].UnsignedAttributes | Where-Object { $_.Oid.Value -eq '1.2.840.113549.1.9.16.2.14' })) {
            throw 'The package signature has no timestamp.'
        }
    }
}
finally {
    if (Test-Path -LiteralPath $unpacked) { Remove-Item -LiteralPath $unpacked -Recurse -Force }
}
Write-Host "Release files for $Version are complete$(if ($ExpectedSignerThumbprint) { ' and signed' })."
