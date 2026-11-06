param(
    [Parameter(Mandatory = $true)] [string] $ComfyUIRoot,
    [Parameter(Mandatory = $true)] [string] $Python,
    [ValidateSet("audit", "full")]
    [string] $Mode = "full",
    [string] $Results = "",
    [int] $Port = 8628,
    [ValidateRange(3, 1000)] [int] $Steps = 8,
    [ValidateSet("0.2.37", "0.2.39")]
    [string] $KitchenImplementation = "0.2.37",
    [string] $LoraName = "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$kit = $PSScriptRoot
$identity = Get-Content (Join-Path $kit "release.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$ComfyUIRoot = (Resolve-Path $ComfyUIRoot).Path
$Python = (Resolve-Path $Python).Path
if ([IO.Path]::GetFileName($LoraName) -ne $LoraName -or
    -not $LoraName.EndsWith('.safetensors', [StringComparison]::OrdinalIgnoreCase)) {
    throw "LoraName must be a .safetensors filename under ComfyUI models/loras"
}
if ([string]::IsNullOrWhiteSpace($Results)) {
    $Results = Join-Path $kit "results-$Mode"
}
$Results = [IO.Path]::GetFullPath($Results)
New-Item -ItemType Directory -Force -Path $Results | Out-Null

& $Python (Join-Path $kit "preflight.py") --report (Join-Path $Results "preflight.json")
if ($LASTEXITCODE -ne 0) { throw "Triton/GPU preflight failed; no benchmark was run" }
& $Python (Join-Path $kit "validate_wheels.py") --wheelhouse (Join-Path $kit "wheelhouse") --report (Join-Path $Results "wheel-validation.json")
if ($LASTEXITCODE -ne 0) { throw "wheel audit failed; no benchmark was run" }
if ($Mode -eq "audit") { Write-Host "PASS $Results"; return }

$versionFile = Join-Path $ComfyUIRoot "comfyui_version.py"
if (-not (Test-Path $versionFile)) { throw "missing ComfyUI version file: $versionFile" }
$versionSource = Get-Content $versionFile -Raw -Encoding UTF8
$versionMatch = [regex]::Match($versionSource, '__version__\s*=\s*["'']([0-9]+\.[0-9]+\.[0-9]+)["'']')
if (-not $versionMatch.Success) { throw "cannot determine ComfyUI version from $versionFile" }
$comfyVersion = $versionMatch.Groups[1].Value
$supportedComfySeries = @('0.38', '0.39', '0.40', '0.42', '0.43', '0.44')
$comfySeries = ($comfyVersion -split '\.')[0..1] -join '.'
if ($supportedComfySeries -notcontains $comfySeries) {
    throw "ComfyUI $comfyVersion is outside this kit's validated 0.38.x-0.40.x/0.42.x-0.44.x set; update the kit before benchmarking"
}
Write-Host "ComfyUI $comfyVersion; isolated Kitchen implementation $KitchenImplementation"
$pattern = "spark_h3_kitchen-$KitchenImplementation+spark.h3.$($identity.architecture).$($identity.cuda_tag).1-*-win_amd64.whl"
$wheels = @(Get-ChildItem (Join-Path $kit "wheelhouse") -Filter $pattern)
if ($wheels.Count -ne 1) { throw "expected one spark-h3-kitchen wheel for implementation $KitchenImplementation; found $($wheels.Count)" }
& $Python -c "import requests, websocket"
if ($LASTEXITCODE -ne 0) { throw "Python requires requests and websocket-client" }
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    throw "full validation requires ffprobe in PATH"
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "full validation requires ffmpeg in PATH"
}

$env:COMFYUI_ROOT = $ComfyUIRoot
$env:SPARK_WHEEL = $wheels[0].FullName
$env:SPARK_RESULTS = $Results
$env:SPARK_TEST_PORT = "$Port"
$env:SPARK_STEPS = "$Steps"
$env:SPARK_LORA_NAME = $LoraName
& $Python (Join-Path $kit "run_lightx2v_three_variants.py")
if ($LASTEXITCODE -ne 0) { throw "$Mode validation failed; see $Results" }
Write-Host "PASS $Results"
