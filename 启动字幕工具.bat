@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "CONDA_ENV_NAME=whisper-env"
set "CONDA_BAT="
set "PYTHON_EXE="

for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
    if not defined CONDA_BAT set "CONDA_BAT=%%I"
)

if not defined CONDA_BAT (
    echo Conda was not found on PATH.
    echo This launcher requires: conda activate "%CONDA_ENV_NAME%"
    pause
    exit /b 1
)

call "%CONDA_BAT%" activate "%CONDA_ENV_NAME%"
if errorlevel 1 (
    echo Failed to activate Conda env: %CONDA_ENV_NAME%
    echo Conda entry point: %CONDA_BAT%
    pause
    exit /b 1
)

if not defined CONDA_PREFIX (
    echo Conda activation did not set CONDA_PREFIX.
    echo Expected env: %CONDA_ENV_NAME%
    pause
    exit /b 1
)

set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Activated env Python was not found:
    echo   %PYTHON_EXE%
    pause
    exit /b 1
)

set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Library\usr\bin;%CONDA_PREFIX%\DLLs;%PATH%"
if exist "%CONDA_PREFIX%\Library\lib\tcl8.6\init.tcl" set "TCL_LIBRARY=%CONDA_PREFIX%\Library\lib\tcl8.6"
if exist "%CONDA_PREFIX%\Library\lib\tk8.6\tk.tcl" set "TK_LIBRARY=%CONDA_PREFIX%\Library\lib\tk8.6"

echo Launching GUI with %PYTHON_EXE%
echo Activated Conda env: %CONDA_ENV_NAME%
echo CONDA_PREFIX=%CONDA_PREFIX%
if defined TCL_LIBRARY echo TCL_LIBRARY=%TCL_LIBRARY%
if defined TK_LIBRARY echo TK_LIBRARY=%TK_LIBRARY%
"%PYTHON_EXE%" "%~dp0subtitle_gui.py"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Launch failed with exit code %EXIT_CODE%.
    echo Activated Conda env: %CONDA_ENV_NAME%
    echo CONDA_PREFIX=%CONDA_PREFIX%
    echo Python used: %PYTHON_EXE%
    echo Check that tkinter, whisper, and torch are installed in this exact environment.
    pause
)

exit /b %EXIT_CODE%
