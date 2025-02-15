# Suppresses warnings.
$WarningPreference = 'SilentlyContinue'
# This instructs PowerShell to treat non-terminating errors as terminating errors, which will halt script execution.
$ErrorActionPreference = "Stop"

$path_input = "./hello.cu"
$path_output = Join-Path (Split-Path $path_input -Parent) "bin" (Split-Path $path_input -LeafBase)

if (-not (Test-Path $path_output) -or ((Get-Item $path_input).CreationTime -ge (Get-Item $path_output).CreationTime)) {
    Write-Host "Compiling '$path_input' into '$path_output'"
    nvcc `
    -arch=sm_89 `
    -D=NDEBUG `
    -g -G `
    -dopt=on `
    -restrict `
    -I="$Env:HOME/ThunderKittens/include" `
    -maxrregcount=255 `
    -std=c++20 `
    -expt-relaxed-constexpr `
    -D__CUDA_NO_HALF_CONVERSIONS__ `
    -o (New-Item $path_output -Force) `
    $path_input
} else {
    Write-Host "The '$path_output' is up to date."
}

<#
pwsh build.ps1
#>
