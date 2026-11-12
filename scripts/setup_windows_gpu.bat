@echo off
REM ============================================================================
REM  VSLM - Windows GPU install
REM
REM  Installs a torch build whose CUDA version bitsandbytes actually ships a
REM  binary for, without letting pip replace anything else in the environment.
REM
REM  Why this script exists:
REM
REM    bitsandbytes 0.50.2 (the latest release) ships native binaries for CUDA
REM    11.8 through 13.0. The PyPI build of torch is compiled against CUDA 13.2,
REM    for which no binary exists. Importing bitsandbytes succeeds; the failure
REM    arrives later, inside model loading.
REM
REM    Do NOT run `pip install --force-reinstall bitsandbytes`. That pulls torch
REM    from PyPI along with it and reinstalls every dependency at its newest
REM    version, which breaks unrelated packages in the environment.
REM
REM  Run it from anywhere:  scripts\setup_windows_gpu.bat
REM ============================================================================

setlocal

REM run from the repository root even if the script is double-clicked
cd /d "%~dp0.."

set TORCH_VER=2.12.1
set TV_VER=0.27.1
set CUDA_TAG=cu130
set INDEX=https://download.pytorch.org/whl/%CUDA_TAG%

echo.
echo ============================================================
echo  VSLM - Windows GPU install
echo ============================================================
echo.
echo [1/4] torch %TORCH_VER%+%CUDA_TAG% and torchvision %TV_VER%+%CUDA_TAG%
echo       %INDEX%
echo       (cu130 is the newest CUDA release bitsandbytes ships a binary for)
python -m pip install --index-url %INDEX% ^
    torch==%TORCH_VER%+%CUDA_TAG% torchvision==%TV_VER%+%CUDA_TAG%
if errorlevel 1 goto fail

echo.
echo [2/4] bitsandbytes, with --no-deps so pip cannot pull torch from PyPI
python -m pip install --no-deps bitsandbytes==0.50.2
if errorlevel 1 goto fail

echo.
echo [3/4] restoring the two packages a --force-reinstall bumped
python -m pip install "fsspec[http]==2026.7.0" "filelock>=3.24.2,<4"
if errorlevel 1 goto fail

echo.
echo [4/4] the rest of the training stack
echo       (torch and bitsandbytes are not in the extras, so pip leaves them alone)
python -m pip install -e ".[train,dev]"
if errorlevel 1 goto fail

echo.
echo verifying...
python -c "import torch, bitsandbytes as bnb; print('torch     ', torch.__version__); print('cuda build', torch.version.cuda); print('gpu       ', torch.cuda.get_device_name(0)); t = torch.zeros(16, 16, device='cuda', dtype=torch.bfloat16); bnb.functional.quantize_4bit(t, quant_type='nf4'); print('4-bit quantisation works')"
if errorlevel 1 goto fail

echo.
echo ============================================================
echo  done. next:
echo      make preflight
echo      make data
echo      make train
echo ============================================================
exit /b 0

:fail
echo.
echo ============================================================
echo  FAILED - read the message above and fix it before continuing.
echo ============================================================
exit /b 1
