# The build script for the .cu files.
# Gets used in the VS Code tasks.

param (
    [Parameter(Mandatory)][string]$Path,
    [switch]$BuildOnly
)

$WarningPreference = 'SilentlyContinue'; $ErrorActionPreference = "Stop"; Set-StrictMode -Version Latest

if ((Split-Path $Path -Extension) -ne ".cu") {
    throw "Expected a .cu file as the build target."
}

$path_bin = Join-Path (Split-Path $Path -Parent) "bin"
$path_output = Join-Path (New-Item $path_bin -ItemType Directory -Force) (Split-Path $Path -LeafBase)

if (-not (Test-Path $path_output) -or 
    ($(Get-Item $Path).LastWriteTime -ge $(Get-Item $path_output).LastWriteTime) -or 
    ($(Get-Item $PSCommandPath).LastWriteTime -ge $(Get-Item $path_output).LastWriteTime)) {
    Write-Host "Compiling '$Path' into '$path_output'"

    $cpp_libs_includes = 
        @(
            # "Crow/include"
            "cpp_libs/xoshiro/include"
            "cpp_libs/cpp-httplib"
            "cpp_libs/json/include"
            "cpp_libs/eigen"
            "hopfield_dictionary/host_cpp_lib"
        ) | ForEach-Object { "-I$_" }
    $cpp_host_files =
        @(
            "hopfield_dictionary/host_cpp_lib/hopfield_dict.cpp" # For performance the host side Hopfield dictionary implementation has been factored into its own library.
        )
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
        "-std", "c++20" # Compiles with the selected C++ standard
        $Path, "cpu_math.o"
        "-o", $path_output # The output path
    )

    g++ -O3 -std=c++20 $cpp_libs_includes -c $cpp_host_files -o cpu_math.o
    if (-not $?) { throw "g++ compilation failed" }
    nvcc $nvcc_args 
} else {
    # Write-Host "The '$path_output' is up to date."
}

if ($? -and (-not $BuildOnly)){ # Runs the executable if the compilation was successful or if it is already up to date.
    Set-Location $path_bin
    Write-Host "Running: $path_output"
    & $path_output 2>&1 | Tee-Object -FilePath "$path_output.log"
    if (-not $?) { throw "The program execution failed." }
}
