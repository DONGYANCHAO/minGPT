@echo off
REM minGPT Development Environment Setup Script for Windows

echo ========================================
echo   minGPT Development Environment Setup
echo ========================================
echo.

REM Check Python
echo [1/6] Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

echo.
echo [2/6] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo       Virtual environment created at .venv
) else (
    echo       Virtual environment already exists
)

echo.
echo [3/6] Activating virtual environment...
call .venv\Scripts\activate.bat
echo       Virtual environment activated

echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [5/6] Installing package in development mode...
pip install -e ".[dev]"

echo.
echo [6/6] Installing pre-commit hooks...
pre-commit install
pre-commit install --hook-type pre-push

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To activate the virtual environment, run:
echo   .venv\Scripts\activate.bat
echo.
echo Available commands:
echo   pytest                  - Run tests
echo   pytest --cov=mingpt     - Run tests with coverage
echo   black .                 - Format code
echo   isort .                 - Sort imports
echo   ruff check .            - Lint code
echo   mypy mingpt/            - Type check
echo   pre-commit run --all-files - Run all hooks
echo.
echo To create a new release:
echo   bump2version patch     - Bug fixes (0.0.1 -^> 0.0.2)
echo   bump2version minor     - New features (0.0.1 -^> 0.1.0)
echo   bump2version major     - Breaking changes (0.0.1 -^> 1.0.0)
echo.
pause
