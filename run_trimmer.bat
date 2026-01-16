@echo off
REM Batch file to launch Trimmer application
REM Tries uv first, falls back to activating venv manually

where uv >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Launching Trimmer with uv...
    uv run trimmer
) else (
    echo uv not found, activating virtual environment...
    if exist ".venv\Scripts\activate.bat" (
        call .venv\Scripts\activate.bat
        python -m trimmer.basic_trimmer_class
        call .venv\Scripts\deactivate.bat
    ) else (
        echo Error: No virtual environment found at .venv
        echo Please run: uv venv and uv sync
        pause
        exit /b 1
    )
)
