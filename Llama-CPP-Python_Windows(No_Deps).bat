@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo Llama-CPP-Python Auto-Installer (Windows / No-Deps)
echo =======================================================
echo.

:: 1. DEFINE ALL PROVIDED LINKS IN A LOOKUP TABLE
:: Pytorch + CU13
set "LINK_cp310_cu130=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu130-Basic-win-20260315/llama_cpp_python-0.3.33+cu130.basic-cp310-cp310-win_amd64.whl"
set "LINK_cp311_cu130=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu130-Basic-win-20260315/llama_cpp_python-0.3.33+cu130.basic-cp311-cp311-win_amd64.whl"
set "LINK_cp312_cu130=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu130-Basic-win-20260315/llama_cpp_python-0.3.33+cu130.basic-cp312-cp312-win_amd64.whl"
set "LINK_cp313_cu130=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu130-Basic-win-20260315/llama_cpp_python-0.3.33+cu130.basic-cp313-cp313-win_amd64.whl"
set "LINK_cp314_cu130=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu130-Basic-win-20260315/llama_cpp_python-0.3.33+cu130.basic-cp314-cp314-win_amd64.whl"

:: Pytorch + CU12.8
set "LINK_cp310_cu128=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu128-Basic-win-20260315/llama_cpp_python-0.3.33+cu128.basic-cp310-cp310-win_amd64.whl"
set "LINK_cp311_cu128=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu128-Basic-win-20260315/llama_cpp_python-0.3.33+cu128.basic-cp311-cp311-win_amd64.whl"
set "LINK_cp312_cu128=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu128-Basic-win-20260315/llama_cpp_python-0.3.33+cu128.basic-cp312-cp312-win_amd64.whl"
set "LINK_cp313_cu128=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu128-Basic-win-20260315/llama_cpp_python-0.3.33+cu128.basic-cp313-cp313-win_amd64.whl"
set "LINK_cp314_cu128=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu128-Basic-win-20260315/llama_cpp_python-0.3.33+cu128.basic-cp314-cp314-win_amd64.whl"

:: Pytorch + CU12.6
set "LINK_cp310_cu126=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu126-Basic-win-20260315/llama_cpp_python-0.3.33+cu126.basic-cp310-cp310-win_amd64.whl"
set "LINK_cp311_cu126=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu126-Basic-win-20260315/llama_cpp_python-0.3.33+cu126.basic-cp311-cp311-win_amd64.whl"
set "LINK_cp312_cu126=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu126-Basic-win-20260315/llama_cpp_python-0.3.33+cu126.basic-cp312-cp312-win_amd64.whl"
set "LINK_cp313_cu126=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu126-Basic-win-20260315/llama_cpp_python-0.3.33+cu126.basic-cp313-cp313-win_amd64.whl"
set "LINK_cp314_cu126=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu126-Basic-win-20260315/llama_cpp_python-0.3.33+cu126.basic-cp314-cp314-win_amd64.whl"

:: Pytorch + CU12.4
set "LINK_cp310_cu124=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu124-Basic-win-20260315/llama_cpp_python-0.3.33+cu124.basic-cp310-cp310-win_amd64.whl"
set "LINK_cp311_cu124=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu124-Basic-win-20260315/llama_cpp_python-0.3.33+cu124.basic-cp311-cp311-win_amd64.whl"
set "LINK_cp312_cu124=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu124-Basic-win-20260315/llama_cpp_python-0.3.33+cu124.basic-cp312-cp312-win_amd64.whl"
set "LINK_cp313_cu124=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu124-Basic-win-20260315/llama_cpp_python-0.3.33+cu124.basic-cp313-cp313-win_amd64.whl"
set "LINK_cp314_cu124=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu124-Basic-win-20260315/llama_cpp_python-0.3.33+cu124.basic-cp314-cp314-win_amd64.whl"


:: 2. LOCATE PYTHON
set "PYTHON_EXE=%~dp0..\..\..\python_embeded\python.exe"

for %%i in ("%PYTHON_EXE%") do set "PYTHON_EXE=%%~fi"

if exist "%PYTHON_EXE%" (
    echo [INFO] Found ComfyUI Portable Python at: 
    echo "%PYTHON_EXE%"
) else (
    echo [WARNING] ComfyUI Portable Python not found. 
    echo [INFO] Falling back to global system Python...
    set "PYTHON_EXE=python"
)
echo.


:: 3. WRITE TEMPORARY PYTHON SCRIPT 
set "TMP_PY=%~dp0temp_detect.py"
set "TMP_OUT=%~dp0temp_out.txt"

echo import sys> "%TMP_PY%"
echo try:>> "%TMP_PY%"
echo     import torch>> "%TMP_PY%"
echo     py='cp'+str(sys.version_info.major)+str(sys.version_info.minor)>> "%TMP_PY%"
echo     c_ver='cu'+torch.version.cuda.replace('.','') if torch.version.cuda else 'cpu'>> "%TMP_PY%"
echo     print(py+'^|'+c_ver)>> "%TMP_PY%"
echo except Exception as e:>> "%TMP_PY%"
echo     print('ERROR^|ERROR')>> "%TMP_PY%"


:: 4. EXECUTE SCRIPT AND SAVE OUTPUT TO TEXT FILE
echo [INFO] Querying system for Python and CUDA versions...
"%PYTHON_EXE%" "%TMP_PY%" > "%TMP_OUT%"


:: 5. READ TEXT FILE AND EXTRACT VARIABLES
for /f "usebackq tokens=1,2 delims=|" %%a in ("%TMP_OUT%") do (
    set PY_VER=%%a
    set CUDA_VER=%%b
)

:: Clean up the temp files silently
if exist "%TMP_PY%" del "%TMP_PY%"
if exist "%TMP_OUT%" del "%TMP_OUT%"

:: Safety checks
if "%PY_VER%"=="ERROR" (
    echo [ERROR] PyTorch is not installed or failed to load.
    echo Please install PyTorch first so we can detect your CUDA version.
    pause
    exit /b
)

if "%PY_VER%"=="" (
    echo [ERROR] Failed to extract environment variables.
    pause
    exit /b
)

:: Standardize CUDA 13.0 output just in case
if "%CUDA_VER%"=="cu13" set "CUDA_VER=cu130"

echo -----------------------------------------
echo Detected Environment:
echo Python Version:  %PY_VER%
echo CUDA Version:    %CUDA_VER%
echo OS Platform:     win_amd64
echo -----------------------------------------
echo.


:: 6. MATCH DETECTED SYSTEM TO YOUR LINKS
call set "TARGET_WHEEL=%%LINK_%PY_VER%_%CUDA_VER%%%"

if "%TARGET_WHEEL%"=="" (
    echo [ERROR] No pre-built wheel found for Python %PY_VER% with CUDA %CUDA_VER%.
    echo The available versions are Python 3.10-3.14 with CUDA 12.4, 12.6, 12.8, or 13.0.
    pause
    exit /b
)

echo [INFO] Match found! Downloading and installing without dependencies:
echo %TARGET_WHEEL%
echo.


:: 7. EXECUTE INSTALLATION (Now with --no-deps and --force-reinstall)
"%PYTHON_EXE%" -m pip install --no-deps --force-reinstall "%TARGET_WHEEL%"

echo.
echo =======================================================
echo [SUCCESS] Llama-CPP-Python Setup Complete!
echo =======================================================
pause