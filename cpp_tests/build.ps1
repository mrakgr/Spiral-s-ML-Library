# Suppresses warnings.
$WarningPreference = 'SilentlyContinue'
# This instructs PowerShell to treat non-terminating errors as terminating errors, which will halt script execution.
$ErrorActionPreference = "Stop"

nvcc `
    -arch=sm_90a `
    -D=NDEBUG `
    -g -G `
    -dopt=on `
    -restrict `
    -I="$Env:HOME/ThunderKittens/include" `
    -maxrregcount=255 `
    -std=c++20 `
    -expt-relaxed-constexpr `
    -D__CUDA_NO_HALF_CONVERSIONS__ `
    -o hello.out `
    hello.cu

<#
pwsh build.ps1
#>
