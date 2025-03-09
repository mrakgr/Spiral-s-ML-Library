$WarningPreference = 'SilentlyContinue'; $ErrorActionPreference = "Stop"; Set-StrictMode -Version Latest

$path_script = "./build.ps1" 
$path_input = "./test0.cu"
$path_output = Join-Path (New-Item "./bin" -ItemType Directory -Force) (Split-Path $path_input -LeafBase)

if (-not (Test-Path $path_output) -or 
    ($(Get-Item $path_input).LastWriteTime -ge $(Get-Item $path_output).LastWriteTime) -or 
    ($(Get-Item $path_script).LastWriteTime -ge $(Get-Item $path_output).LastWriteTime)) {
    Write-Host "Compiling '$path_input' into '$path_output'"
    nvcc `
        -arch=native `
        -D=NDEBUG `
        -g -G `
        -dopt=on `
        -restrict `
        -I="$Env:HOME/ThunderKittens/include" `
        -maxrregcount=255 `
        -std=c++20 `
        -expt-relaxed-constexpr `
        -D__CUDA_NO_HALF_CONVERSIONS__ `
        -diag-suppress 550,20012,68,39,177 `
        -o $path_output `
        $path_input
} else {
    # Write-Host "The '$path_output' is up to date."
}

if ($?){
    & $path_output
}


<#
cd /mnt/c/Spiral_s_ML_Library/cpp_cuda
pwsh build.ps1
#>
