$WarningPreference = 'SilentlyContinue'; $ErrorActionPreference = "Stop"; Set-StrictMode -Version Latest

$path_input = "./hello.cu"
$path_output_dir = Join-Path (Split-Path $path_input -Parent) "bin"
if (-not (Test-Path $path_output_dir)) {
    New-Item $path_output_dir -ItemType Directory
}
$path_output = Join-Path $path_output_dir (Split-Path $path_input -LeafBase)

if (-not (Test-Path $path_output) -or ((Get-Item $path_input).CreationTime -ge (Get-Item $path_output).CreationTime)) {
    Write-Host "Compiling '$path_input' into '$path_output'"
    nvcc `
    -arch=sm_120 `
    -D=NDEBUG `
    -g -G `
    -dopt=on `
    -restrict `
    -I="$Env:HOME/ThunderKittens/include" `
    -maxrregcount=255 `
    -std=c++20 `
    -expt-relaxed-constexpr `
    -D__CUDA_NO_HALF_CONVERSIONS__ `
    -o $path_output `
    $path_input
} else {
    Write-Host "The '$path_output' is up to date."
}

& "$path_output"

<#
pwsh build.ps1
#>
