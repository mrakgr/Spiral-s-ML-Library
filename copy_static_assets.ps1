# Copies the shoelace assets into the static/bundles folder.
# Shoelace is the web UI library used in some games.
# https://shoelace.style/

$sourcePath = "node_modules/@shoelace-style/shoelace/dist/assets/"
$destinationPath = "ui/backend/static/bundles/shoelace/assets/"

if (-not (Test-Path $destinationPath)) {
    Copy-Item -Path $sourcePath -Destination $destinationPath -Force -Recurse
}

# The following line, also works and would be more roboust, but its Windows only.
# Robocopy $sourcePath $destinationPath /mir

# symlinks didn't work for me.