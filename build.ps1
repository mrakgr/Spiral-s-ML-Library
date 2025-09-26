# The build script for the .cu files.
# Gets used the VS Code tasks.

param (
    [Parameter(Mandatory)][string]$Path,
    [switch]$BuildOnly
)

$WarningPreference = 'SilentlyContinue'; $ErrorActionPreference = "Stop"; Set-StrictMode -Version Latest

$path_bin = Join-Path (Split-Path $Path -Parent) "bin"
$path_output = Join-Path (New-Item $path_bin -ItemType Directory -Force) (Split-Path $Path -LeafBase)

if (-not (Test-Path $path_output) -or 
    ($(Get-Item $Path).LastWriteTime -ge $(Get-Item $path_output).LastWriteTime) -or 
    ($(Get-Item $PSCommandPath).LastWriteTime -ge $(Get-Item $path_output).LastWriteTime)) {
    Write-Host "Compiling '$Path' into '$path_output'"
    nvcc `
    -arch=sm_120a `
    -D=NDEBUG `
    -g -G `
    -dopt=on `
    -restrict `
    -expt-relaxed-constexpr `
    -D__CUDA_NO_HALF_CONVERSIONS__ `
    -diag-suppress 550,20012,68,39,177 `
    -std=c++20 `
    -o $path_output `
    $Path
} else {
    # Write-Host "The '$path_output' is up to date."
}

if ($? -and (-not $BuildOnly)){ # Runs the executable if the compilation was successful or if it is already up to date.
    Set-Location $path_bin
    Write-Host "Running: $path_output"
    & $path_output
}

<#
pwsh build.ps1 -Path cpp_cuda/test1.cu
#>