# The build script for the .cpp files.
# Gets used in the VS Code tasks.

param (
    [Parameter(Mandatory)][string]$Path,
    [switch]$BuildOnly
)

$WarningPreference = 'SilentlyContinue'; $ErrorActionPreference = "Stop"; Set-StrictMode -Version Latest

if ((Split-Path $Path -Extension) -ne ".cpp") {
    throw "Expected a .cpp file as the build target."
}

$path_bin = Join-Path (Split-Path $Path -Parent) "bin"
$path_output_bin = Join-Path (New-Item $path_bin -ItemType Directory -Force) (Split-Path $Path -LeafBase)
$path_output_obj = "$path_output_bin.o"

if (-not (Test-Path $path_output_bin) -or 
    ($(Get-Item $Path).LastWriteTime -ge $(Get-Item $path_output_bin).LastWriteTime) -or 
    ($(Get-Item $PSCommandPath).LastWriteTime -ge $(Get-Item $path_output_bin).LastWriteTime)) {
    Write-Host "Compiling '$Path' into '$path_output_bin'"

    $cpp_libs_includes = 
        @(
            # "Crow/include"
            "cpp_libs/xoshiro/include"
            "cpp_libs/cpp-httplib"
            "cpp_libs/json/include"
            "cpp_libs/eigen"
        ) | ForEach-Object { "-I$_" }
    $nvcc_args = @(
        $cpp_libs_includes
        # "-D", "NDEBUG" # Turns off the asserts
        "-Xcompiler", "-Wno-format-zero-length" # Suppresses print("") statement warnings
        "-arch", "sm_120a" # The cuda architecture
        "-g" # Generates the debug info on host
        "-G" # Generates the debug info on device
        "-dopt", "on" # Turns on the device optimizations
        "-Xcompiler", "-O3" # Turns on the host optimizations
        "-restrict" # Turns on the restricted pointer optimizations
        "-expt-relaxed-constexpr" # Allows relaxed constant expressions
        "-D__CUDA_NO_HALF_CONVERSIONS__" # Hack to compile Cutlass with half float types
        "-diag-suppress", "550,20012,68,39,177" # Suppresses various warnings
        "-std=c++20" # Compiles with the selected C++ standard
        "-c", "$([System.IO.Path]::ChangeExtension($Path, 'cu'))" # Input file
        "-o", $path_output_obj # The output object path
        )
    $gpp_args = @(
        $cpp_libs_includes
        "-I/usr/local/cuda/include"
        "-O3" # Turns on the host optimizations
        "-std=c++20" # Compiles with the selected C++ standard
        "-Wno-format-zero-length" # Suppresses print("") statement warnings
        "-Wno-attributes" # Suppresses Cuda attribute warnings.
        $Path, $path_output_obj # The input files
        "-L/usr/local/cuda/lib64", "-lcudart" # Cuda runtime library links
        "-o", $path_output_bin # The output binary path.
    )

    
    if (Test-Path $path_output_bin) { Remove-Item $path_output_bin }
    echo "Compiling with nvcc..." && nvcc $nvcc_args && echo "Compiling with g++..." && g++ $gpp_args
    if (Test-Path $path_output_obj) { Remove-Item $path_output_obj }
} else {
    # Write-Host "The '$path_output_bin' is up to date."
}

if ((Test-Path $path_output_bin) -and (-not $BuildOnly)){ # Runs the executable if the compilation was successful or if it is already up to date.
    Set-Location $path_bin
    Write-Host "Running: $path_output_bin"
    & $path_output_bin 2>&1 | Tee-Object -FilePath "$path_output_bin.log"
    if (-not $?) { throw "The program execution failed." }
}
