#!/usr/bin/env python3
"""
版本同步检查脚本

用于 pre-commit 钩子，确保以下文件中的版本号保持一致:
    - pyproject.toml
    - mingpt/__init__.py
    - README.md (徽章)
"""

import re
import sys
from pathlib import Path


def get_version_from_pyproject() -> str | None:
    """从 pyproject.toml 获取版本"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return None

    try:
        import tomllib

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # 尝试从 project 部分获取版本
        version = data.get("project", {}).get("version")
        if version:
            return version

        # 尝试从 setuptools-scm 获取
        if "setuptools-scm" in data.get("build-system", {}).get("requires", []):
            return "scm"  # 动态版本管理

    except ImportError:
        # Python < 3.11, 使用 tomli
        try:
            import tomli

            with open(pyproject_path, "rb") as f:
                data = tomli.load(f)

            version = data.get("project", {}).get("version")
            if version:
                return version

        except ImportError:
            pass
    except Exception:
        pass

    return None


def get_version_from_init() -> str | None:
    """从 mingpt/__init__.py 获取版本"""
    init_path = Path("mingpt/__init__.py")
    if not init_path.exists():
        return None

    try:
        content = init_path.read_text(encoding="utf-8")

        # 查找 __version__ = "x.x.x"
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)

    except Exception:
        pass

    return None


def get_version_from_readme() -> str | None:
    """从 README.md 获取版本徽章版本"""
    readme_path = Path("README.md")
    if not readme_path.exists():
        return None

    try:
        content = readme_path.read_text(encoding="utf-8")

        # 查找 PyPI 版本徽章
        match = re.search(
            r'https://img\.shields\.io/pypi/v/mingpt[^?]*\?version=([^\s&\)]+)',
            content,
        )
        if match:
            return match.group(1)

        # 查找其他版本徽章格式
        match = re.search(
            r'https://img\.shields\.io/pypi/v/mingpt[^\s\)]*',
            content,
        )
        if match:
            return "badge_present"  # 徽章存在但无法解析版本

    except Exception:
        pass

    return None


def check_version_consistency() -> list[str]:
    """检查版本一致性"""
    errors = []

    pyproject_version = get_version_from_pyproject()
    init_version = get_version_from_init()
    readme_version = get_version_from_readme()

    # 如果使用了 setuptools-scm，跳过版本检查
    if pyproject_version == "scm":
        print("使用 setuptools-scm 进行动态版本管理，跳过版本一致性检查")
        return []

    versions = {
        "pyproject.toml": pyproject_version,
        "mingpt/__init__.py": init_version,
        "README.md": readme_version,
    }

    # 过滤出有效的版本号
    valid_versions = {
        k: v for k, v in versions.items() if v is not None and v != "badge_present"
    }

    if not valid_versions:
        errors.append("未找到任何版本信息")
        return errors

    # 检查版本是否一致
    unique_versions = set(valid_versions.values())
    if len(unique_versions) > 1:
        errors.append("版本号不一致:")
        for file, version in valid_versions.items():
            errors.append(f"  - {file}: {version}")

    # 检查是否有文件缺少版本
    for file, version in versions.items():
        if version is None:
            errors.append(f"{file}: 未找到版本信息")

    return errors


def main() -> int:
    """主函数"""
    print("检查版本同步...")

    errors = check_version_consistency()

    if errors:
        print("发现以下问题:")
        for error in errors:
            print(f"  - {error}")
        print("\n提示: 请确保所有文件中的版本号保持一致")
        return 1

    print("版本同步检查通过！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
