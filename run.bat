@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "PYTHON_BIN="
where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_BIN=py -3"
) else (
    where python >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_BIN=python"
    ) else (
        where python3 >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_BIN=python3"
        )
    )
)

if "%PYTHON_BIN%"=="" (
    echo Python was not found on PATH. Install Python 3 and try again.
    exit /b 1
)

echo Installing Python dependencies...
%PYTHON_BIN% -m pip install -r "%SCRIPT_DIR%\app\requirements.txt"
if errorlevel 1 exit /b %errorlevel%

echo Starting unified app at http://localhost:8000 ...
echo Driving demo: http://localhost:8000/driving/ or ../examples
echo Tuning dashboard: http://localhost:8000/tuning/
pushd "%SCRIPT_DIR%"
%PYTHON_BIN% -m app.app
set "APP_EXIT=%errorlevel%"
popd

exit /b %APP_EXIT%
