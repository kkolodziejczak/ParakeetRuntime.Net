<#
.SYNOPSIS
Walks the owner through the manual acceptance checks of a release build before it is tagged.

.DESCRIPTION
Builds the release files for -Version (scripts/Build-Release.ps1 and Test-ReleaseArtifacts.ps1), then runs
every check: it fires the commands, says what only a person can do (speak, unplug, look at the browser),
checks what it can on its own, and asks for a verdict. Enter = pass, f = fail (then type or dictate what you
saw), s = skip, r = run the step again. Each verdict is written at once to
artifacts\acceptance\<version>\results-<time>.md. Start again at any step with -From 3.1.

Everything it changes is put back: the models folder it moves aside is restored, and every process it starts
is stopped, also when you press Ctrl+C in this window. The audio is FLEURS clips in .\audio (see README.md).
-PrepareOnly builds the release and the test files, prints their paths and stops.

.EXAMPLE
.\scripts\acceptance\Start-Acceptance.ps1 -Version 0.6.0
.\scripts\acceptance\Start-Acceptance.ps1 -Version 0.6.0 -SkipBuild -From 3.1
#>
param(
    [Parameter(Mandatory)] [ValidatePattern('^\d+\.\d+\.\d+$')] [string] $Version,
    [string] $From,
    [switch] $SkipBuild,
    [switch] $PrepareOnly
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Release = Join-Path $Repo "artifacts\release\$Version"
$Exe = Join-Path $Release "ParakeetRuntime.Net-$Version-win-x64\parakeet-runtime.exe"
$Package = Join-Path $Release "ParakeetRuntime.Net.$Version.nupkg"
$Work = Join-Path $Repo "artifacts\acceptance\$Version"
$Results = Join-Path $Work ("results-{0:yyyy-MM-dd-HHmm}.md" -f (Get-Date))
$Models = Join-Path $env:LOCALAPPDATA 'ParakeetRuntime\models'
$ModelFile = 'parakeet-tdt-0.6b-v3.q8_0.gguf'
$Aside = "$Models.acceptance-aside"
$Origin = 'http://127.0.0.1:5092'
$Clips = Import-Csv -LiteralPath (Join-Path $PSScriptRoot 'audio\clips.tsv') -Delimiter "`t" -Encoding UTF8
$script:Started = New-Object System.Collections.Generic.List[System.Diagnostics.Process]
$script:Reached = [string]::IsNullOrWhiteSpace($From)
$script:Passed = 0; $script:Failed = 0; $script:Skipped = 0

function Say([string] $text, [string] $color = 'Gray') { Write-Host $text -ForegroundColor $color }
function Do-This([string] $text) { Write-Host "  YOU: $text" -ForegroundColor Yellow }
function Expect([string] $text) { Write-Host "  EXPECTED: $text" -ForegroundColor Cyan }
function Good([string] $text) { Write-Host "  AUTO-CHECK OK: $text" -ForegroundColor Green }
function Bad([string] $text) { Write-Host "  AUTO-CHECK FAILED: $text" -ForegroundColor Red }
function Wait-You([string] $text = 'Press Enter when ready') { [void](Read-Host "  $text") }
function Ask-YesNo([string] $question) { (Read-Host "  $question [y/N]") -match '^(y|yes|t|tak)$' }

function Clip([string] $file) { Join-Path $PSScriptRoot "audio\$file" }
function Reference([string] $file) { ($Clips | Where-Object file -eq $file).text }

function Quote-All([string[]] $arguments) { ($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' ' }

function Run-Exe([string[]] $arguments, [string] $stdoutFile) {
    Say "  > parakeet-runtime $($arguments -join ' ')" DarkGray
    $options = @{ FilePath = $Exe; ArgumentList = (Quote-All $arguments); NoNewWindow = $true; Wait = $true; PassThru = $true }
    if ($stdoutFile) { $options.RedirectStandardOutput = $stdoutFile }
    $process = Start-Process @options
    Say "  (exit code $($process.ExitCode))" DarkGray
    $process.ExitCode
}

function Transcribe-Clip([string] $file, [string[]] $extra = @()) {
    $out = Join-Path $Work "$file.txt"
    Run-Exe (@('transcribe', (Clip $file)) + $extra) $out | Out-Null
    Say "  GOT:       $((Get-Content -LiteralPath $out -Raw -Encoding UTF8).Trim())" White
    Say "  REFERENCE: $(Reference $file)" White
}

function Write-Result([string] $id, [string] $title, [string] $verdict, [string] $note) {
    if (-not (Test-Path $Results)) {
        $build = (& $Exe --version 2>&1 | Out-String).Trim()
        @("# Acceptance results, $(Get-Date -Format 'yyyy-MM-dd HH:mm')", '', "Build: parakeet-runtime $build", '',
          '| Step | Check | Result | Note |', '| --- | --- | --- | --- |') | Set-Content -LiteralPath $Results -Encoding UTF8
    }
    "| $id | $title | $verdict | $($note -replace '\|', '/') |" | Add-Content -LiteralPath $Results -Encoding UTF8
}

function Step([string] $id, [string] $title, [scriptblock] $body) {
    if (-not $script:Reached) {
        if ($id -ne $From) { return }
        $script:Reached = $true
    }
    while ($true) {
        Write-Host ''
        Say "=== $id  $title" White
        try { & $body | Out-Host } catch { Say "  The step stopped: $($_.Exception.Message)" Red }
        $answer = Read-Host '  Verdict: Enter = pass, f = fail, s = skip, r = run again'
        switch -Regex ($answer) {
            '^r' { continue }
            '^s' { $script:Skipped++; Write-Result $id $title 'skipped' ''; return }
            '^f' {
                $note = Read-Host '  What did you see? (type or dictate)'
                $script:Failed++; Write-Result $id $title '**FAIL**' $note; return
            }
            default { $script:Passed++; Write-Result $id $title 'pass' ''; return }
        }
    }
}

function Move-ModelsAside {
    if (Test-Path $Aside) { throw "$Aside is left from an earlier run; rename it back to $Models first." }
    if (Test-Path $Models) { Rename-Item -LiteralPath $Models -NewName (Split-Path -Leaf $Aside) }
    Say "  (moved the models folder aside to $Aside)" DarkGray
}

function Restore-Models {
    if (-not (Test-Path $Aside)) { return }
    if (Test-Path $Models) {
        $downloaded = "$Models.acceptance-downloaded-$(Get-Date -Format yyyyMMddHHmmss)"
        Rename-Item -LiteralPath $Models -NewName (Split-Path -Leaf $downloaded)
        if (Ask-YesNo "Delete the test download in $downloaded (714 MB)?") { Remove-Item -LiteralPath $downloaded -Recurse -Force }
        else { Say "  (the test download stays in $downloaded)" DarkGray }
    }
    Rename-Item -LiteralPath $Aside -NewName (Split-Path -Leaf $Models)
    Say '  (put the models folder back)' DarkGray
}

function Start-Exe([string[]] $arguments, [switch] $Visible) {
    $options = @{ FilePath = $Exe; PassThru = $true; WindowStyle = $(if ($Visible) { 'Normal' } else { 'Minimized' }) }
    if ($arguments) { $options.ArgumentList = Quote-All $arguments }
    $process = Start-Process @options
    $script:Started.Add($process)
    $process
}

function Wait-Health([int] $seconds = 90, [switch] $AnyAnswer) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        $code = & curl.exe -s -o NUL -w '%{http_code}' "$Origin/health"
        if ($code -eq '200' -or ($AnyAnswer -and $code -ne '000')) { return $true }
        Start-Sleep -Milliseconds 500
    }
    Bad "the server did not answer within $seconds s."
    $false
}

function Ensure-Server {
    if (-not (Port-Listeners)) { Start-Exe @('serve', '--no-browser') | Out-Null; [void](Wait-Health) }
}

function Stop-Started {
    foreach ($process in $script:Started) {
        if (-not $process.HasExited) { & taskkill.exe /PID $process.Id /T /F 2>&1 | Out-Null }
    }
    $script:Started.Clear()
}

function Port-Listeners { @(Get-NetTCPConnection -LocalPort 5092 -State Listen -ErrorAction SilentlyContinue) }

function Pcm-Of([string] $path) {
    $bytes = [IO.File]::ReadAllBytes($path)
    $at = 12
    while ($at + 8 -le $bytes.Length) {
        $id = [Text.Encoding]::ASCII.GetString($bytes, $at, 4)
        $size = [BitConverter]::ToInt32($bytes, $at + 4)
        if ($id -eq 'data') {
            $pcm = New-Object byte[] $size
            [Array]::Copy($bytes, $at + 8, $pcm, 0, $size)
            return , $pcm
        }
        $at += 8 + $size + ($size % 2)
    }
    throw "$path has no data chunk."
}

function Write-Wav([string] $path, [byte[]] $pcm) {
    $writer = New-Object IO.BinaryWriter ([IO.File]::Create($path))
    try {
        $writer.Write([Text.Encoding]::ASCII.GetBytes('RIFF')); $writer.Write([int](36 + $pcm.Length)); $writer.Write([Text.Encoding]::ASCII.GetBytes('WAVEfmt '))
        $writer.Write([int]16); $writer.Write([int16]1); $writer.Write([int16]1); $writer.Write([int]16000); $writer.Write([int]32000); $writer.Write([int16]2); $writer.Write([int16]16)
        $writer.Write([Text.Encoding]::ASCII.GetBytes('data')); $writer.Write([int]$pcm.Length); $writer.Write($pcm)
    }
    finally { $writer.Close() }
}

function New-TestFiles {
    $long = Join-Path $Work 'english-6-minutes.wav'
    if (-not (Test-Path $long)) {
        $halfSecond = New-Object byte[] 16000
        $pieces = New-Object System.Collections.Generic.List[byte]
        $english = @($Clips | Where-Object language -eq 'en_us' | ForEach-Object { , (Pcm-Of (Clip $_.file)) })
        for ($i = 0; $pieces.Count -lt 32000 * 360; $i++) { $pieces.AddRange([byte[]]$english[$i % $english.Count]); $pieces.AddRange($halfSecond) }
        Write-Wav $long $pieces.ToArray()
    }
    $tooLong = Join-Path $Work 'eleven-minutes-of-silence.wav'
    if (-not (Test-Path $tooLong)) { Write-Wav $tooLong (New-Object byte[] (32000 * 660)) }
    $fake = Join-Path $Work 'an-mp3-renamed.wav'
    [IO.File]::WriteAllBytes($fake, [byte[]](0x49, 0x44, 0x33, 0x04, 0, 0, 0, 0, 0, 0x21) + (1..4000 | ForEach-Object { [byte]($_ % 251) }))
    @{ Long = $long; TooLong = $tooLong; Fake = $fake }
}

$previousOutputEncoding = [Console]::OutputEncoding
[Console]::OutputEncoding = New-Object Text.UTF8Encoding $false
New-Item -ItemType Directory -Force -Path $Work | Out-Null

if (-not $SkipBuild) {
    Say "Building the $Version release files..." White
    & (Join-Path $Repo 'scripts\Build-Release.ps1') -Version $Version | Out-Null
    & (Join-Path $Repo 'scripts\Test-ReleaseArtifacts.ps1') -Version $Version
}
if (-not (Test-Path $Exe)) { throw "No release build at $Exe; run without -SkipBuild." }
$Files = New-TestFiles
$en = 'en-01.wav'
$pl = 'pl-01.wav'
if ($PrepareOnly) {
    $Files.Values | ForEach-Object { Say "  $_ ($([math]::Round((Get-Item $_).Length / 32000, 1)) s of 16 kHz audio)" }
    [Console]::OutputEncoding = $previousOutputEncoding
    return
}

Say "ParakeetRuntime.Net $Version acceptance. Results go to:" White
Say "  $Results"
Say 'Keep this window and the browser side by side. Ctrl+C here stops the wizard and puts everything back.'

try {
    # ---------------------------------------------------------------- 1 Microphone
    Step '1.1' 'Record English speech' {
        Do-This "When it says ""Recording"", read this aloud, then press Enter: ""$(Reference 'en-03.wav')"""
        Run-Exe @('record') | Out-Null
        Expect 'the printed text matches what you read.'
    }
    Step '1.2' 'Record Polish speech' {
        Do-This "Read this aloud, then press Enter: ""$(Reference 'pl-01.wav')"""
        Run-Exe @('record') | Out-Null
        Expect 'the printed text matches, with Polish letters.'
    }
    Step '1.3' 'Unplug the microphone while recording' {
        Start-Process 'ms-settings:sound'
        Do-This 'Start talking; while it records, unplug the microphone (or turn it off in the Sound settings I opened), then press Enter.'
        Run-Exe @('record') | Out-Null
        Expect '"The microphone stopped working: The microphone was unplugged or turned off."'
        Do-This 'Plug it back in, or turn it back on.'
        Wait-You
    }
    Step '1.4' 'Microphone access off for desktop apps' {
        Start-Process 'ms-settings:privacy-microphone'
        Do-This 'Turn OFF "Let desktop apps access your microphone" in the window I opened.'
        Wait-You
        Run-Exe @('record') | Out-Null
        Expect 'the message names that privacy setting.'
        Do-This 'Turn it back ON.'
        Wait-You
    }
    Step '1.5' 'No microphone at all' {
        Start-Process 'ms-settings:sound'
        Do-This 'Disable every input device in Sound settings (or skip this step with s).'
        Wait-You
        Run-Exe @('record') | Out-Null
        Expect '"Windows has no microphone to record from."'
        Do-This 'Enable your microphone again.'
        Wait-You
    }
    Step '1.6' 'Silence' {
        Do-This 'Stay silent for 3 seconds, then press Enter.'
        Run-Exe @('record') | Out-Null
        Expect 'nothing, or a stray word such as "Yeah." Note what it wrote if you mark it.'
    }

    # ---------------------------------------------------------------- 2 Exe commands
    Step '2.1' '--help lists the commands' {
        $help = & $Exe --help 2>&1 | Out-String
        $help | Out-Host
        $missing = @('transcribe', 'record', 'download', 'serve') | Where-Object { $help -notmatch $_ }
        if ($missing) { Bad "missing from --help: $($missing -join ', ')" } else { Good 'transcribe, record, download and serve are listed.' }
    }
    Step '2.2' 'Transcribe speech' {
        foreach ($file in @($Clips | Where-Object language -eq 'en_us' | Select-Object -First 3 -ExpandProperty file)) { Transcribe-Clip $file }
        Expect 'each GOT line says what its REFERENCE says (punctuation and numbers may differ).'
    }
    Step '2.3' 'Transcribe 6 minutes, cut every 20 s' {
        Run-Exe @('transcribe', $Files.Long) | Out-Null
        Expect "the six clips' sentences repeat in order with none missing; the clips are:"
        $Clips | Where-Object language -eq 'en_us' | ForEach-Object { Say "    $($_.text)" }
    }
    Step '2.4' 'An MP3 renamed to .wav' {
        $code = Run-Exe @('transcribe', $Files.Fake)
        if ($code -eq 1) { Good 'exit code 1.' } else { Bad "exit code $code, expected 1." }
        Expect 'a clear message that it is not a WAV file.'
    }
    Step '2.5' 'No model, then download' {
        Move-ModelsAside
        try {
            Run-Exe @('transcribe', (Clip $en)) | Out-Null
            Expect 'it tells you to run "parakeet-runtime download".'
            if (Ask-YesNo 'Now download the model (714 MB)?') {
                Run-Exe @('download') | Out-Null
                Expect 'progress while downloading, then "installed".'
            }
        }
        finally { Restore-Models }
    }
    Step '2.6' 'Install from a copy, offline' {
        Move-ModelsAside
        try {
            Do-This 'Optional: turn Wi-Fi off now to prove it needs no network.'
            Wait-You
            Run-Exe @('download', '--from', (Join-Path $Aside $ModelFile)) | Out-Null
            Expect 'it checks the copy and installs it.'
            Do-This 'Turn Wi-Fi back on if you turned it off.'
            Wait-You
        }
        finally { Restore-Models }
    }
    Step '2.7' 'Ctrl+C during download leaves no .part file' {
        Move-ModelsAside
        try {
            $download = Start-Exe @('download') -Visible
            Do-This 'In the new window, wait until the progress moves, then press Ctrl+C there.'
            Wait-You 'Press Enter here after that window has stopped'
            if (-not $download.HasExited) { & taskkill.exe /PID $download.Id /T /F 2>&1 | Out-Null }
            Run-Exe @('download', '--from', (Join-Path $Aside $ModelFile)) | Out-Null
            $parts = @(Get-ChildItem $Models -Filter '*.part' -ErrorAction SilentlyContinue)
            if ($parts.Count -eq 0) { Good 'no .part file left after the next run.' } else { Bad "left behind: $($parts.Name -join ', ')" }
            Expect 'the window said it stopped.'
        }
        finally { Restore-Models }
    }
    Step '2.8' 'Polish letters on screen and in a file' {
        Transcribe-Clip $pl
        Start-Process notepad.exe (Join-Path $Work "$pl.txt")
        Expect 'ą ć ę ł ń ó ś ź ż look right here and in Notepad (the transcript went to a file through stdout).'
    }

    # ---------------------------------------------------------------- 3 HTTP server
    Step '3.1' 'curl gets JSON with text and duration' {
        Ensure-Server
        & curl.exe -s -F "file=@$(Clip $en)" "$Origin/v1/audio/transcriptions"; Write-Host ''
        Say "  REFERENCE: $(Reference $en)" White
        Expect '{"text":"...","duration":6.5...} with that text.'
    }
    Step '3.2' 'response_format=text' {
        Ensure-Server
        & curl.exe -s -F "file=@$(Clip $en)" -F response_format=text "$Origin/v1/audio/transcriptions"; Write-Host ''
        Expect 'only the text, no JSON.'
    }
    Step '3.3' 'Only 127.0.0.1 listens; no firewall prompt' {
        Ensure-Server
        $addresses = @(Port-Listeners | ForEach-Object LocalAddress | Sort-Object -Unique)
        if ($addresses.Count -eq 1 -and $addresses[0] -eq '127.0.0.1') { Good 'listening on 127.0.0.1 only.' } else { Bad "listening on $($addresses -join ', ')" }
        Expect 'Windows Firewall asked nothing.'
    }
    Step '3.4' 'An OpenAI-style request' {
        Ensure-Server
        & curl.exe -s -F "file=@$(Clip $en)" -F model=whisper-1 -F language=en -F temperature=0 "$Origin/v1/audio/transcriptions"; Write-Host ''
        Expect 'the extra OpenAI fields are ignored and the text comes back. Optional: point an OpenAI SDK or Postman at http://127.0.0.1:5092/v1.'
    }
    Stop-Started
    Step '3.5' 'No model: /health and the console' {
        Move-ModelsAside
        try {
            Start-Exe @('serve', '--no-browser') -Visible | Out-Null
            [void](Wait-Health -AnyAnswer)
            $health = & curl.exe -s "$Origin/health"
            $health | Out-Host
            if ($health -match '"no-model"') { Good 'status is no-model.' } else { Bad 'status is not no-model.' }
            Expect 'the JSON has the URL, size and folder; the new window names "parakeet-runtime download".'
            Wait-You 'Press Enter after reading the new window'
        }
        finally { Stop-Started; Restore-Models }
    }
    Step '3.6' 'Ctrl+C stops the server and frees the port' {
        $server = Start-Exe @('serve', '--no-browser') -Visible
        [void](Wait-Health)
        Do-This 'Press Ctrl+C in the server window.'
        Wait-You 'Press Enter here after that'
        [void]$server.WaitForExit(15000)
        if (-not (Port-Listeners)) { Good 'port 5092 is free.' } else { Bad 'port 5092 is still taken.' }
        Expect 'the window printed "Stopped."'
    }
    Stop-Started

    # ---------------------------------------------------------------- 4 Page
    Step '4.1' 'Double-click opens the page; --no-browser does not' {
        Start-Exe @('serve', '--no-browser') -Visible | Out-Null
        Expect 'a console window and NO browser tab.'
        Wait-You 'Press Enter after checking no tab opened'
        Stop-Started
        Start-Exe -Visible | Out-Null
        Expect 'a console window says where it serves, and the browser opens http://127.0.0.1:5092/.'
    }
    Step '4.2' 'The pill and the version' {
        Expect "the pill reads ""Ready · GPU: <your discrete GPU> (D3D12)"", never an integrated GPU, and the version is $Version."
    }
    Step '4.3' 'Drop a WAV of speech, copy the text' {
        Start-Process explorer.exe (Join-Path $PSScriptRoot 'audio')
        Start-Process notepad.exe
        Do-This "Drop $en from the folder I opened on the page. Press Copy text and paste into Notepad."
        Say "  REFERENCE: $(Reference $en)" White
        Expect '"Transcribing en-01.wav...", then the text, audio length and time; the pasted text matches.'
    }
    Step '4.4' 'Choose file with the keyboard only' {
        Do-This 'Tab to "Choose file", press Enter, pick en-02.wav, no mouse.'
        Expect 'the same result as dropping it.'
    }
    Step '4.5' 'Bad files' {
        Start-Process explorer.exe $Work
        Do-This 'From the folder I opened, drop an-mp3-renamed.wav, then eleven-minutes-of-silence.wav, then any picture.'
        Expect 'a red box in plain words for each, and the page still works after.'
    }
    Step '4.6' 'Copy the curl line and run it' {
        Do-This "Copy the curl line from the page, paste it into a terminal opened in $(Join-Path $PSScriptRoot 'audio'), change the file name to en-01.wav, run it."
        Expect 'the same text comes back in the terminal.'
    }
    Stop-Started
    Step '4.7' 'Model missing, download from the page' {
        Move-ModelsAside
        try {
            Start-Exe -Visible | Out-Null
            Expect 'the page shows "The speech model is not installed" with size, source and folder.'
            if (Ask-YesNo 'Download from the page now (714 MB)?') {
                Do-This 'Press "Download model"; refresh the tab mid-download; wait for "Ready".'
                Expect 'the bar fills, a refresh shows the same progress, then "Loading model..." and "Ready".'
                Wait-You 'Press Enter when it is Ready'
            }
        }
        finally { Stop-Started; Restore-Models }
    }
    Step '4.8' 'Server closed while the page is open' {
        Start-Exe -Visible | Out-Null
        Do-This 'When the page is ready, close the console window, then drop a WAV on the page.'
        Expect '"The server is not answering..."'
    }
    Stop-Started
    Step '4.9' 'Phone width' {
        Start-Exe -Visible | Out-Null
        Do-This 'Narrow the browser window to phone width (or F12, device toolbar).'
        Expect 'nothing cut off, no sideways scrolling, the curl line wraps.'
    }
    Step '4.10' 'Edge and Firefox' {
        Start-Process 'msedge.exe' "$Origin/" -ErrorAction SilentlyContinue
        if (Get-Command firefox.exe -ErrorAction SilentlyContinue) { Start-Process firefox.exe "$Origin/" }
        Do-This 'In each: drop a WAV (4.3) and copy the curl line (4.6).'
    }
    Stop-Started

    # ---------------------------------------------------------------- 5 Device choice and the package
    Step '5.1' 'The default GPU is discrete; an integrated one only when named' {
        Run-Exe @('--list-gpus') | Out-Null
        Transcribe-Clip $en
        Expect '"Transcribing ... on GPU: <a discrete GPU>", or the CPU on a machine without one.'
        Do-This 'If the list shows an integrated GPU (Intel UHD, Iris, Radeon Graphics), type part of its name; else press Enter.'
        $integrated = Read-Host '  Integrated GPU name'
        if ($integrated) { Transcribe-Clip $en @('--device', "gpu:$integrated"); Expect "... on GPU: <the $integrated GPU>." }
    }
    Step '5.2' 'The package in a new console app' {
        $app = Join-Path $Work 'package-app'
        if (Test-Path $app) { Remove-Item $app -Recurse -Force }
        & dotnet new console -f net10.0 -o $app 2>&1 | Out-Null
        $project = Join-Path $app 'package-app.csproj'
        (Get-Content $project -Raw) -replace '<TargetFramework>net10.0</TargetFramework>', '<TargetFramework>net10.0-windows</TargetFramework>' | Set-Content $project -Encoding UTF8
        "<configuration><packageSources><clear /><add key=""release"" value=""$Release"" /><add key=""nuget.org"" value=""https://api.nuget.org/v3/index.json"" /></packageSources></configuration>" |
            Set-Content (Join-Path $app 'nuget.config') -Encoding UTF8
        @"
using ParakeetRuntime.Net;

var model = SpeechModel.Default();
await model.EnsureInstalledAsync();
using var engine = await SpeechEngine.LoadAsync(model);
string text = await engine.TranscribeWavAsync(@"$(Clip $en)");
Console.WriteLine(text);
"@ | Set-Content (Join-Path $app 'Program.cs') -Encoding UTF8
        $previousCache = $env:NUGET_PACKAGES
        $env:NUGET_PACKAGES = Join-Path $Work 'nuget-cache'
        Push-Location $app
        try {
            & dotnet add package ParakeetRuntime.Net --version $Version 2>&1 | Select-Object -Last 1 | Out-Host
            $printed = (& dotnet run 2>&1 | Out-String).Trim()
            Say "  GOT:       $printed" White
            Say "  REFERENCE: $(Reference $en)" White
        }
        finally { Pop-Location; $env:NUGET_PACKAGES = $previousCache }
        Expect 'the README sample built against the release package prints the clip''s text. (A private NuGet cache keeps this unsigned build out of your real one.)'
    }
    Step '5.3' 'The package readme' {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip = [IO.Compression.ZipFile]::OpenRead($Package)
        $readme = Join-Path $Work 'package-README.md'
        try { [IO.Compression.ZipFileExtensions]::ExtractToFile(($zip.Entries | Where-Object FullName -eq 'README.md'), $readme, $true) }
        finally { $zip.Dispose() }
        if (Get-Command code -ErrorAction SilentlyContinue) { & code $readme } else { Start-Process notepad.exe $readme }
        Expect 'it reads well as the nuget.org page: install, sample, how to use it, speed, requirements, license.'
    }
}
finally {
    Stop-Started
    Restore-Models
    [Console]::OutputEncoding = $previousOutputEncoding
    Write-Host ''
    Say "Done: $script:Passed passed, $script:Failed failed, $script:Skipped skipped. Results: $Results" White
    if ($script:Failed -eq 0 -and $script:Passed -gt 0) {
        Say "Next, once .github/release-notes/v$Version.md is on master: git tag -a v$Version -m ""ParakeetRuntime.Net $Version""; git push origin v$Version" White
    }
}
