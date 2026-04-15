# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Modern `pyproject.toml` configuration replacing `setup.py`
- Pre-commit hooks configuration with black, isort, ruff, mypy
- GitHub Actions CI/CD pipeline with multi-version testing
- PyPI automated publishing workflow
- Security scanning with bandit, safety, and pip-audit
- Code coverage reporting with codecov
- Development setup scripts (`setup.sh` and `setup.bat`)
- Comprehensive `.gitignore` for Python projects
- MANIFEST.in for package distribution

### Changed

- Updated `mingpt/__init__.py` with proper version management
- Enhanced README with badges and development documentation

## [0.0.1] - 2024-01-01

### Added

- Initial release
- GPT model implementation in PyTorch
- Byte Pair Encoder (BPE)
- Training utilities
- Demo notebooks
- Example projects (adder, chargpt)
