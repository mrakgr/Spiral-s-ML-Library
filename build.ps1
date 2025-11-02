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

$path_cu_file = "$([System.IO.Path]::ChangeExtension($Path, 'cu'))"
$path_bin_dir = Join-Path (Split-Path $Path -Parent) "bin"
$path_output_bin_file = Join-Path (New-Item $path_bin_dir -ItemType Directory -Force) (Split-Path $Path -LeafBase)
$path_output_obj_file = "$path_output_bin_file.o"

if (-not (Test-Path $path_output_bin_file) -or 
    ($(Get-Item $Path).LastWriteTime -ge $(Get-Item $path_output_bin_file).LastWriteTime) -or 
    ($(Get-Item $PSCommandPath).LastWriteTime -ge $(Get-Item $path_output_bin_file).LastWriteTime)) {
    Write-Host "Compiling '$Path' into '$path_output_bin_file'"

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
        "-c", $path_cu_file # Input file
        "-o", $path_output_obj_file # The output object path
        )
    $gpp_args = @(
        $cpp_libs_includes
        "-O3" # Turns on the host optimizations
        "-std=c++20" # Compiles with the selected C++ standard
        "-Wno-format-zero-length" # Suppresses print("") statement warnings
        "-Wno-attributes" # Suppresses Cuda attribute warnings.
        $(if (Test-Path $path_cu_file) { @( # Cuda specific compilation options. It's structured like this so we can compile arbitrary .cpp files even without an accompanying .cu one.
            "-I/usr/local/cuda/include" # Cuda include
            $path_output_obj_file # The Cuda object file from the previous compilation step
            "-L/usr/local/cuda/lib64", "-lcudart" # Cuda runtime library links
        )}) 
        $Path # The Cpp host input file
        "-o", $path_output_bin_file # The output binary path.
    )
    
    if (Test-Path $path_output_bin_file) { Remove-Item $path_output_bin_file }
    if (Test-Path $path_output_obj_file) { Remove-Item $path_output_obj_file }
    if (Test-Path $path_cu_file) { Write-Output "Compiling with nvcc..."; nvcc $nvcc_args } 
    if ($?) { Write-Output "Compiling with g++..."; g++ $gpp_args }
    if (Test-Path $path_output_obj_file) { Remove-Item $path_output_obj_file }
} else {
    # Write-Host "The '$path_output_bin_file' is up to date."
}

if ((Test-Path $path_output_bin_file) -and (-not $BuildOnly)){ # Runs the executable if the compilation was successful or if it is already up to date.
    Set-Location $path_bin_dir
    Write-Host "Running: $path_output_bin_file"
    & $path_output_bin_file 2>&1 | Tee-Object -FilePath "$path_output_bin_file.log"
    if (-not $?) { throw "The program execution failed." }
}
