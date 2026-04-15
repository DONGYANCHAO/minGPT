#!/usr/bin/env python3
"""
minGPT Pre-commit 一键安装脚本

此脚本用于自动安装和配置 pre-commit 钩子，确保代码提交前自动进行质量检查。

使用方法:
    python scripts/setup_precommit.py

功能:
    1. 检查 Python 版本 (>=3.8)
    2. 安装 pre-commit 工具
    3. 安装项目开发依赖
    4. 安装 git hooks
    5. 运行首次检查（可选）
    6. 验证安装结果

作者: minGPT Team
许可证: MIT
"""

import os
import subprocess
import sys
from pathlib import Path


# 颜色配置
class Colors:
    """终端颜色输出"""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(message: str) -> None:
    """打印标题"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD} {message}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_success(message: str) -> None:
    """打印成功消息"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    """打印警告消息"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """打印错误消息"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_info(message: str) -> None:
    """打印信息消息"""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


def check_python_version() -> bool:
    """检查 Python 版本是否 >= 3.8"""
    print_info("检查 Python 版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python 版本 {version.major}.{version.minor} 不兼容，需要 >= 3.8")
        return False


def run_command(
    command: list[str],
    description: str,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess | None:
    """
    运行 shell 命令

    Args:
        command: 命令列表
        description: 命令描述
        check: 是否检查返回码
        capture_output: 是否捕获输出

    Returns:
        CompletedProcess 对象或 None
    """
    print_info(f"执行: {description}")
    try:
        result = subprocess.run(
            command,
            check=check,
            capture_output=capture_output,
            text=True,
        )
        print_success(f"{description} - 完成")
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"{description} - 失败")
        if e.stdout:
            print(f"输出: {e.stdout}")
        if e.stderr:
            print(f"错误: {e.stderr}")
        if check:
            sys.exit(1)
        return None


def install_precommit_tool() -> bool:
    """安装 pre-commit 工具"""
    print_info("安装 pre-commit 工具...")

    # 检查是否已安装
    result = run_command(
        [sys.executable, "-m", "pip", "show", "pre-commit"],
        "检查 pre-commit 是否已安装",
        check=False,
        capture_output=True,
    )

    if result and result.returncode == 0:
        print_success("pre-commit 已安装")
        return True

    # 安装 pre-commit
    result = run_command(
        [sys.executable, "-m", "pip", "install", "pre-commit"],
        "安装 pre-commit",
        check=False,
    )

    if result and result.returncode == 0:
        print_success("pre-commit 安装成功")
        return True
    else:
        print_error("pre-commit 安装失败")
        return False


def install_dev_dependencies() -> bool:
    """安装开发依赖"""
    print_info("安装开发依赖...")

    dev_packages = [
        "black>=23.0.0",
        "isort>=5.12.0",
        "ruff>=0.1.0",
        "mypy>=1.0.0",
        "bandit[toml]>=1.7.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "nbstripout>=0.6.0",
        "nbqa>=1.7.0",
        "autoflake>=2.0.0",
        "pyupgrade>=3.0.0",
    ]

    result = run_command(
        [sys.executable, "-m", "pip", "install"] + dev_packages,
        "安装开发依赖包",
        check=False,
    )

    if result and result.returncode == 0:
        print_success("开发依赖安装成功")
        return True
    else:
        print_warning("部分开发依赖安装失败，但 pre-commit 仍可工作")
        return True


def install_git_hooks() -> bool:
    """安装 git hooks"""
    print_info("安装 git hooks...")

    # 检查是否在 git 仓库中
    git_dir = Path(".git")
    if not git_dir.exists():
        print_error("当前目录不是 git 仓库，请先运行 'git init'")
        return False

    result = run_command(
        [sys.executable, "-m", "pre_commit", "install"],
        "安装 pre-commit hooks",
        check=False,
    )

    if result and result.returncode == 0:
        print_success("Git hooks 安装成功")
        return True
    else:
        print_error("Git hooks 安装失败")
        return False


def install_commit_msg_hook() -> bool:
    """安装 commit-msg hook（可选）"""
    print_info("安装 commit-msg hook...")

    result = run_command(
        [sys.executable, "-m", "pre_commit", "install", "--hook-type", "commit-msg"],
        "安装 commit-msg hook",
        check=False,
    )

    if result and result.returncode == 0:
        print_success("Commit-msg hook 安装成功")
        return True
    else:
        print_warning("Commit-msg hook 安装失败（可选）")
        return True


def run_first_check() -> bool:
    """运行首次检查（可选）"""
    print_info("是否运行首次检查？这将检查所有文件（可能需要几分钟）")
    response = input("运行首次检查? [y/N]: ").strip().lower()

    if response in ("y", "yes"):
        print_info("运行 pre-commit 检查所有文件...")
        result = run_command(
            [sys.executable, "-m", "pre_commit", "run", "--all-files"],
            "首次检查",
            check=False,
        )

        if result and result.returncode == 0:
            print_success("首次检查通过")
            return True
        else:
            print_warning("首次检查发现问题，请根据提示修复")
            return True
    else:
        print_info("跳过首次检查")
        return True


def validate_installation() -> bool:
    """验证安装结果"""
    print_info("验证安装...")

    # 检查 pre-commit 配置
    config_file = Path(".pre-commit-config.yaml")
    if not config_file.exists():
        print_error("未找到 .pre-commit-config.yaml 配置文件")
        return False
    print_success("找到 pre-commit 配置文件")

    # 检查 git hooks
    hook_file = Path(".git/hooks/pre-commit")
    if not hook_file.exists():
        print_error("Git pre-commit hook 未安装")
        return False
    print_success("Git pre-commit hook 已安装")

    # 验证配置
    result = run_command(
        [sys.executable, "-m", "pre_commit", "validate-config"],
        "验证 pre-commit 配置",
        check=False,
        capture_output=True,
    )

    if result and result.returncode == 0:
        print_success("Pre-commit 配置有效")
        return True
    else:
        print_warning("Pre-commit 配置验证失败")
        return True


def print_usage() -> None:
    """打印使用说明"""
    print_header("Pre-commit 使用说明")

    usage = f"""
{Colors.BOLD}基本用法:{Colors.ENDC}

  1. 自动运行（提交时）:
     {Colors.OKCYAN}git commit -m "your message"{Colors.ENDC}
     提交前会自动运行所有钩子

  2. 手动运行所有检查:
     {Colors.OKCYAN}pre-commit run --all-files{Colors.ENDC}

  3. 运行特定钩子:
     {Colors.OKCYAN}pre-commit run black{Colors.ENDC}

  4. 跳过钩子（不推荐）:
     {Colors.OKCYAN}git commit -m "your message" --no-verify{Colors.ENDC}

{Colors.BOLD}常用命令:{Colors.ENDC}

  • 更新钩子版本:
    {Colors.OKCYAN}pre-commit autoupdate{Colors.ENDC}

  • 卸载钩子:
    {Colors.OKCYAN}pre-commit uninstall{Colors.ENDC}

  • 查看已安装的钩子:
    {Colors.OKCYAN}pre-commit list{Colors.ENDC}

  • 清理缓存:
    {Colors.OKCYAN}pre-commit clean{Colors.ENDC}

{Colors.BOLD}自动修复:{Colors.ENDC}

  大多数钩子（black、isort、ruff 等）支持自动修复问题。
  如果提交失败，请先查看修改，然后重新提交:

    {Colors.OKCYAN}git add -A{Colors.ENDC}
    {Colors.OKCYAN}git commit -m "your message"{Colors.ENDC}

{Colors.BOLD}故障排除:{Colors.ENDC}

  • 如果钩子运行缓慢，可以跳过特定钩子:
    SKIP=hook-name git commit -m "message"

  • 查看详细日志:
    pre-commit run --verbose

  • 仅检查暂存区文件:
    pre-commit run
"""
    print(usage)


def main() -> int:
    """主函数"""
    print_header("minGPT Pre-commit 安装脚本")

    # 检查 Python 版本
    if not check_python_version():
        return 1

    # 安装 pre-commit 工具
    if not install_precommit_tool():
        return 1

    # 安装开发依赖
    install_dev_dependencies()

    # 安装 git hooks
    if not install_git_hooks():
        return 1

    # 安装 commit-msg hook
    install_commit_msg_hook()

    # 验证安装
    if not validate_installation():
        return 1

    # 运行首次检查（可选）
    run_first_check()

    # 打印使用说明
    print_usage()

    print_header("安装完成！")
    print_success("Pre-commit 已成功配置")
    print_info("现在每次提交代码时都会自动运行质量检查")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
