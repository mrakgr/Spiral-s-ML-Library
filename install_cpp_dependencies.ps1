$WarningPreference = 'SilentlyContinue'; $ErrorActionPreference = "Stop"; Set-StrictMode -Version Latest

$lib_dir = New-Item "$PSScriptRoot/cpp_libs" -ItemType Directory -Force

function Install-Thunderkittens {
    cd $lib_dir
    if (-not (Test-Path ThunderKittens)) {
        Write-Information "Installing ThunderKittens..."
        git clone https://github.com/HazyResearch/ThunderKittens &&
        cd ThunderKittens &&
        git checkout d69697a3337e31d0060178c9049f1184e7e7ad7f # Apr 16, 2025 
    }
}

Install-Thunderkittens

<#
pwsh install_cpp_dependencies.ps1
#>