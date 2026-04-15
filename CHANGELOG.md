# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 完整的工程化配置
  - Pre-commit 预提交钩子配置
  - PyPI 自动发布工作流
  - GitHub Actions CI/CD 流水线
  - 多版本 Python 测试矩阵 (3.8-3.12)
  - 代码覆盖率报告
  - 安全扫描 (Bandit, Safety, pip-audit)
- 现代 Python 项目配置
  - 从 setup.py 迁移到 pyproject.toml
  - 使用 setuptools-scm 进行动态版本管理
  - 完整的项目元数据和依赖配置
- 代码质量工具
  - Black 代码格式化
  - isort import 排序
  - Ruff 代码检查
  - MyPy 类型检查
  - Bandit 安全扫描

### Changed

- 升级项目结构以符合现代 Python 包标准
- 优化依赖管理，支持可选依赖组

### Fixed

- 修复类型提示问题

## [0.0.1] - 2020-XX-XX

### Added

- 初始版本发布
- 实现 GPT 模型核心功能
- 添加 BPE 编码器
- 添加训练器模块
- 提供 adder 和 chargpt 示例项目

---

## Release Notes Template

### Added

- New features

### Changed

- Changes in existing functionality

### Deprecated

- Soon-to-be removed features

### Removed

- Now removed features

### Fixed

- Bug fixes

### Security

- Security improvements

---

## Version Format

- **MAJOR**: Incompatible API changes
- **MINOR**: Added functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

## Tags

- [Unreleased]: Changes since last release
- [YANKED]: Releases that should not be used
