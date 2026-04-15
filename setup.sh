#!/usr/bin/env bash

set -e

echo "========================================"
echo "  minGPT Development Environment Setup"
echo "========================================"
echo ""

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "[1/6] Checking Python version..."
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "      Python version: $PYTHON_VERSION"

echo ""
echo "[2/6] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
    echo "      Virtual environment created at .venv"
else
    echo "      Virtual environment already exists"
fi

echo ""
echo "[3/6] Activating virtual environment..."
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi
echo "      Virtual environment activated"

echo ""
echo "[4/6] Upgrading pip..."
pip install --upgrade pip

echo ""
echo "[5/6] Installing package in development mode..."
pip install -e ".[dev]"

echo ""
echo "[6/6] Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type pre-push

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate  (Linux/macOS)"
echo "  .venv\\Scripts\\activate     (Windows)"
echo ""
echo "Available commands:"
echo "  pytest                  - Run tests"
echo "  pytest --cov=mingpt     - Run tests with coverage"
echo "  black .                 - Format code"
echo "  isort .                 - Sort imports"
echo "  ruff check .            - Lint code"
echo "  mypy mingpt/            - Type check"
echo "  pre-commit run --all-files - Run all hooks"
echo ""
echo "To create a new release:"
echo "  bump2version patch     - Bug fixes (0.0.1 -> 0.0.2)"
echo "  bump2version minor     - New features (0.0.1 -> 0.1.0)"
echo "  bump2version major     - Breaking changes (0.0.1 -> 1.0.0)"
echo ""
