@echo off
REM ============================================================================
REM  VSLM - put the Hugging Face cache on G:
REM
REM  Why this script exists:
REM
REM  The Hugging Face cache defaults to %USERPROFILE%\.cache\huggingface, which
REM  on this machine is on C:. The 4070 SUPER machine has a small C: and a
REM  large G:, and a 1.7B checkpoint needs about 4 GB. A download that runs
REM  out of room part way through dies with:
REM
REM      RuntimeError: Task error: File reconstruction error:
REM      Internal Writer Error: Background writer channel closed
REM
REM  which reads like a network fault but is a full disk. The fix is to point
REM  the cache at the drive with the room.
REM
REM  This sets HF_HOME for the current shell and adds it to the user
REM  environment, so every later shell - including a new Git Bash - gets it.
REM
REM  Run it from anywhere:  scripts\setup_hf_home.bat
REM ============================================================================

setlocal

set HF_DIR=G:\models
set HF_HOME=%HF_DIR%\huggingface
set HF_HUB_CACHE=%HF_HOME%\hub

echo.
echo ============================================================
echo  VSLM - Hugging Face cache on G:
echo ============================================================
echo.
echo  HF_HOME      = %HF_HOME%
echo  HF_HUB_CACHE = %HF_HUB_CACHE%
echo.

REM --- does G: have room? ---------------------------------------------------
if not exist "G:\" (
    echo  ERROR: G: is not available. Is the drive mounted?
    exit /b 1
)

for /f "tokens=1,2,3" %%a in ('powershell -NoProfile -Command ^
  "(Get-PSDrive G).Used; (Get-PSDrive G).Free"') do (
    set USED=%%a
    set FREE=%%b
)
set /a FREEGB=%FREE%/1073741824
echo  G: free space: %FREEGB% GB
if %FREEGB% LSS 8 (
    echo  ERROR: less than 8 GB free on G:. The cache needs about 5 GB.
    exit /b 1
)

REM --- move anything already downloaded ------------------------------------
set OLD_CACHE=%USERPROFILE%\.cache\huggingface
if exist "%OLD_CACHE%" (
    echo.
    echo  moving the existing cache from C: to G: ...
    if not exist "%HF_HOME%" mkdir "%HF_HOME%"
    robocopy "%OLD_CACHE%" "%HF_HOME%" /E /MOVE /NFL /NDL /NJH /NJS /NP
    if errorlevel 8 (
        echo  WARNING: robocopy reported errors. Check that the files
        echo  arrived before deleting anything by hand.
    ) else (
        echo  moved.
    )
) else (
    echo  no cache on C: to move.
)

if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%"

REM --- make it permanent ----------------------------------------------------
echo.
echo  setting the user environment variables ...
setx HF_HOME "%HF_HOME%" >nul
setx HF_HUB_CACHE "%HF_HUB_CACHE%" >nul

echo.
echo ============================================================
echo  done. The cache now lives on G:.
echo.
echo  In this shell it is already set. New shells will get it from the
echo  user environment. To set it for the current Git Bash too:
echo.
echo      export HF_HOME=G:/models/huggingface
echo      export HF_HUB_CACHE=G:/models/huggingface/hub
echo.
echo  Verify with:
echo.
echo      python scripts/check_attention.py
echo ============================================================
exit /b 0
