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

function Install-Crow {
    cd $lib_dir
    New-Item Crow -ItemType Directory -Force &&
    cd Crow &&
    # sudo dpkg -i Crow-1.2.1-Linux.deb
    sudo dpkg -L Crow-1.2.1-Linux.deb
    if (-not (Test-Path Crow)){
        # wget https://github.com/CrowCpp/Crow/releases/download/v1.2.1.2/Crow-1.2.1-Linux.deb &&
        # sudo apt-get update &&
        # sudo apt-get install gdebi &&
        # sudo gdebi Crow-1.2.1-Linux.deb
    }
}

Install-Thunderkittens
Install-Crow

<#
pwsh install_cpp_dependencies.ps1
#>