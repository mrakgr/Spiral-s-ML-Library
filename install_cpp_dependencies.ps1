$WarningPreference = 'SilentlyContinue'; $ErrorActionPreference = "Stop"; Set-StrictMode -Version Latest

$lib_dir = New-Item "$PSScriptRoot/cpp_libs" -ItemType Directory -Force
Write-Host "The C++ library directory is: $lib_dir"

function Install-Git-Repo {
    param (
        [Parameter(Mandatory)][string]$Url,
        [string]$Commit
    )
    $git_name = Split-Path $Url -LeafBase
    function Get-Repo {
        Write-Host "Installing git repo from: $Url"
        git clone $Url
        if (-not $?) { throw "The git clone failed." }
        Push-Location $git_name 
        try {
            if ($Commit) {
                if ($Commit -ne (git rev-parse HEAD)) {
                    Write-Host "Downgrading repo to commit: $Commit"
                    git -c advice.detachedHead=false checkout $Commit # -c advice.detachedHead=false suppresses the detached head checkout warning.
                    if (-not $?) { throw "The checkout failed." }
                } else {
                    Write-Host "The latest branch matches the commit: $Commit"
                }
            }
        } finally { # Unpops the location even if an exception happened.
            Pop-Location
        }
    }
    function Remove-Repo {
        $git_dir = Join-Path (Get-Location) $git_name
        Write-Host "Removing git repo at: $git_dir"
        rm -rf $git_dir
    }

    Set-Location $lib_dir
    if (-not (Test-Path $git_name)) {
        Get-Repo
    } elseif ($Commit) {
        Push-Location $git_name
        if ($Commit -ne (git rev-parse HEAD)) {
            Pop-Location
            Remove-Repo
            Get-Repo
        } else {
            Write-Host "Skipping update as the commit id is the same: $Url"
            Pop-Location
        }
    } else {
        Write-Host "Skipping update due to no target commit id for url: $Url"
    }
}

$repos = @( # The list of dependencies that need installing.
    @{Url = "https://github.com/nessan/xoshiro"; Commit = "176fa191c8493e4c5cb06a44bc083010664fe39b"}
    @{Url = "https://github.com/yhirose/cpp-httplib"; Commit = "89c932f313c6437c38f2982869beacc89c2f2246"}
    @{Url = "https://github.com/nlohmann/json"; Commit = "55f93686c01528224f448c19128836e7df245f72"}
)

foreach ($repos in $repos) {
    Install-Git-Repo @repos
}
