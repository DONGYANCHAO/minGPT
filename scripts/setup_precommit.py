#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def main():
    project_root = Path(__file__).parent.parent
    print("=" * 60)
    print("Setting up pre-commit hooks for minGPT")
    print("=" * 60)

    print("\n1. Checking pre-commit installation...")
    if not run_command([sys.executable, "-m", "pre-commit", "--version"]):
        print("Installing pre-commit...")
        if not run_command([sys.executable, "-m", "pip", "install", "pre-commit"]):
            print("Failed to install pre-commit")
            return 1

    print("\n2. Installing git hook scripts...")
    if not run_command([sys.executable, "-m", "pre-commit", "install"], cwd=project_root):
        print("Failed to install pre-commit hooks")
        return 1

    print("\n3. Running hooks against all files to ensure everything works...")
    if not run_command([sys.executable, "-m", "pre-commit", "run", "--all-files"], cwd=project_root):
        print("Some hooks found issues that need manual fixing")

    print("\n" + "=" * 60)
    print("✅ Pre-commit hooks setup completed successfully!")
    print("=" * 60)
    print("\nHooks will now run automatically before each commit.")
    print("To run manually: pre-commit run --all-files")
    print("To skip hooks for a commit: git commit --no-verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
