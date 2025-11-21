@echo off
REM Automated llama.cpp Setup for Windows
REM This script clones and builds llama.cpp for GGUF quantization

setlocal enabledelayedexpansion

echo ======================================================================
echo   llama.cpp Setup for OVI GGUF Quantization
echo ======================================================================
echo.

REM Find ComfyUI root (navigate up from tools directory)
cd /d "%~dp0"
cd ..\..\..\

set "COMFYUI_ROOT=%CD%"
echo ComfyUI Root: %COMFYUI_ROOT%
echo.

REM Check if llama.cpp already exists
if exist "%COMFYUI_ROOT%\llama.cpp" (
    echo [WARNING] llama.cpp directory already exists!
    echo.
    choice /C YN /M "Remove existing llama.cpp and reinstall?"
    if errorlevel 2 (
        echo Cancelled by user.
        pause
        exit /b 1
    )
    echo Removing existing llama.cpp...
    rmdir /s /q "%COMFYUI_ROOT%\llama.cpp"
)

REM Check for git
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found! Please install Git:
    echo   https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)
echo [OK] Git found

REM Check for cmake
where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake not found! Please install CMake:
    echo   https://cmake.org/download/
    echo   Or: winget install Kitware.CMake
    echo.
    pause
    exit /b 1
)
echo [OK] CMake found

REM Check for Visual Studio Build Tools (look for MSBuild)
where msbuild >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Visual Studio Build Tools may not be installed!
    echo If build fails, install from:
    echo   https://visualstudio.microsoft.com/downloads/
    echo   Or: winget install Microsoft.VisualStudio.2022.BuildTools
    echo.
    choice /C YN /M "Continue anyway?"
    if errorlevel 2 (
        echo Cancelled by user.
        pause
        exit /b 1
    )
) else (
    echo [OK] MSBuild found
)

echo.
echo ======================================================================
echo   Step 1: Cloning llama.cpp
echo ======================================================================
echo.

git clone https://github.com/ggerganov/llama.cpp.git
if errorlevel 1 (
    echo [ERROR] Failed to clone llama.cpp!
    pause
    exit /b 1
)

cd llama.cpp
echo [OK] llama.cpp cloned successfully

echo.
echo ======================================================================
echo   Step 2: Building with CMake (this may take 5-10 minutes)
echo ======================================================================
echo.

REM Try to use ninja if available for faster builds
where ninja >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Using Ninja build system for faster compilation
    cmake -B build -G Ninja
) else (
    echo [INFO] Using default build system
    cmake -B build
)

if errorlevel 1 (
    echo [ERROR] CMake configuration failed!
    echo.
    echo Troubleshooting:
    echo 1. Ensure Visual Studio Build Tools are installed
    echo 2. Try running from "Developer Command Prompt for VS"
    echo 3. Check CMake output above for specific errors
    pause
    exit /b 1
)

echo [OK] CMake configuration successful

echo.
echo Building... (this will take a few minutes)
cmake --build build --config Release

if errorlevel 1 (
    echo [ERROR] Build failed!
    echo.
    echo Check the output above for errors.
    echo Common fixes:
    echo - Restart in "Developer Command Prompt for VS"
    echo - Reinstall Visual Studio Build Tools
    echo - Try: cmake -B build -G "Visual Studio 17 2022"
    pause
    exit /b 1
)

echo [OK] Build successful!

echo.
echo ======================================================================
echo   Step 3: Verifying Installation
echo ======================================================================
echo.

REM Find quantize executable
set "QUANTIZE_EXE="
if exist "build\bin\Release\quantize.exe" (
    set "QUANTIZE_EXE=build\bin\Release\quantize.exe"
) else if exist "build\bin\quantize.exe" (
    set "QUANTIZE_EXE=build\bin\quantize.exe"
) else if exist "build\Release\bin\quantize.exe" (
    set "QUANTIZE_EXE=build\Release\bin\quantize.exe"
) else if exist "build\quantize.exe" (
    set "QUANTIZE_EXE=build\quantize.exe"
)

if "%QUANTIZE_EXE%"=="" (
    echo [ERROR] quantize.exe not found after build!
    echo Expected locations:
    echo   - build\bin\Release\quantize.exe
    echo   - build\bin\quantize.exe
    echo   - build\quantize.exe
    echo.
    echo Please check build output for errors.
    pause
    exit /b 1
)

echo [OK] Found quantize.exe at: %QUANTIZE_EXE%

REM Test quantize tool
"%QUANTIZE_EXE%" 2>nul
if errorlevel 2 (
    echo [OK] quantize tool is executable
) else (
    echo [WARNING] quantize tool may not be working correctly
)

echo.
echo ======================================================================
echo   Installation Complete!
echo ======================================================================
echo.
echo llama.cpp installed at:
echo   %COMFYUI_ROOT%\llama.cpp
echo.
echo quantize tool located at:
echo   %COMFYUI_ROOT%\llama.cpp\%QUANTIZE_EXE%
echo.
echo You can now use the quantization script:
echo   cd custom_nodes\ComfyUI-Ovi\tools
echo   python quantize_ovi_model.py Ovi-960x960-10s.safetensors
echo.
echo Or run quantize manually:
echo   llama.cpp\%QUANTIZE_EXE% input.gguf output.gguf Q4_K_M
echo.

pause
exit /b 0
