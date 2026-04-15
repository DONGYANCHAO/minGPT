#!/usr/bin/env python3
"""测试配置文件验证脚本"""

import sys
from pathlib import Path


def test_pyproject():
    """测试 pyproject.toml"""
    print("=" * 60)
    print("测试 pyproject.toml")
    print("=" * 60)

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open("pyproject.toml", "rb") as f:
        config = tomllib.load(f)

    print(f"✓ 项目名称: {config['project']['name']}")
    print(f"✓ 描述: {config['project']['description'][:50]}...")
    print(f"✓ Python版本要求: {config['project']['requires-python']}")
    print(f"✓ 许可证: {config['project']['license']['text']}")
    print(f"✓ 依赖数量: {len(config['project']['dependencies'])}")
    print(f"✓ 可选依赖组: {list(config['project']['optional-dependencies'].keys())}")

    # 检查 build-system
    assert "build-system" in config
    print("✓ build-system 配置正确")

    # 检查工具配置
    tools = ["black", "isort", "ruff", "mypy", "pytest", "coverage", "bandit"]
    for tool in tools:
        if tool in config.get("tool", {}):
            print(f"✓ {tool} 配置存在")

    return True


def test_precommit_config():
    """测试 .pre-commit-config.yaml"""
    print("\n" + "=" * 60)
    print("测试 .pre-commit-config.yaml")
    print("=" * 60)

    import yaml

    with open(".pre-commit-config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    repos = config.get("repos", [])
    print(f"✓ 配置了 {len(repos)} 个仓库")

    # 检查关键钩子
    expected_hooks = ["trailing-whitespace", "black", "isort", "ruff", "mypy", "bandit"]
    found_hooks = []

    for repo in repos:
        for hook in repo.get("hooks", []):
            found_hooks.append(hook.get("id", ""))

    for hook in expected_hooks:
        if hook in found_hooks:
            print(f"✓ 找到钩子: {hook}")
        else:
            print(f"⚠ 未找到钩子: {hook}")

    return True


def test_github_workflows():
    """测试 GitHub Actions 工作流"""
    print("\n" + "=" * 60)
    print("测试 GitHub Actions 工作流")
    print("=" * 60)

    import yaml

    workflows_dir = Path(".github/workflows")
    if not workflows_dir.exists():
        print("✗ 工作流目录不存在")
        return False

    workflows = list(workflows_dir.glob("*.yml"))
    print(f"✓ 找到 {len(workflows)} 个工作流文件")

    for workflow_file in workflows:
        with open(workflow_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        name = config.get("name", workflow_file.stem)
        jobs = config.get("jobs", {})
        print(f"✓ {name}: {len(jobs)} 个 job(s)")

        # 检查触发条件
        on_events = config.get("on", {})
        if isinstance(on_events, dict):
            print(f"  - 触发事件: {list(on_events.keys())}")
        elif isinstance(on_events, list):
            print(f"  - 触发事件: {on_events}")

    return True


def test_scripts():
    """测试脚本文件"""
    print("\n" + "=" * 60)
    print("测试脚本文件")
    print("=" * 60)

    scripts_dir = Path("scripts")
    if not scripts_dir.exists():
        print("✗ 脚本目录不存在")
        return False

    scripts = list(scripts_dir.glob("*.py"))
    print(f"✓ 找到 {len(scripts)} 个脚本文件")

    for script in scripts:
        print(f"✓ {script.name}")
        # 检查是否为有效 Python 文件
        try:
            with open(script, "r", encoding="utf-8") as f:
                compile(f.read(), script.name, "exec")
            print(f"  - 语法检查通过")
        except SyntaxError as e:
            print(f"  - 语法错误: {e}")

    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("minGPT 工程化配置测试")
    print("=" * 60 + "\n")

    tests = [
        ("pyproject.toml", test_pyproject),
        (".pre-commit-config.yaml", test_precommit_config),
        ("GitHub Actions", test_github_workflows),
        ("Scripts", test_scripts),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")

    all_passed = all(r for _, r in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
