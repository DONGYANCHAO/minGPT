#!/usr/bin/env python3
"""
Jupyter Notebook 格式检查脚本

用于 pre-commit 钩子，检查 notebook 文件的元数据和格式。

检查项:
    1. 检查 kernel 名称是否标准
    2. 检查是否包含敏感信息
    3. 检查文件大小
    4. 检查 cell 输出是否过大
    5. 检查是否包含绝对路径
"""

import json
import sys
from pathlib import Path


def check_kernel_name(notebook: dict, filepath: Path) -> list[str]:
    """检查 kernel 名称是否标准"""
    errors = []
    metadata = notebook.get("metadata", {})
    kernelspec = metadata.get("kernelspec", {})
    kernel_name = kernelspec.get("name", "")

    # 允许的 kernel 名称
    allowed_kernels = ["python3", "python", "ir", "julia-1.0"]

    if kernel_name and kernel_name not in allowed_kernels:
        errors.append(
            f"{filepath}: 非标准的 kernel 名称 '{kernel_name}'，"
            f"建议使用: {', '.join(allowed_kernels)}"
        )

    return errors


def check_sensitive_info(notebook: dict, filepath: Path) -> list[str]:
    """检查是否包含敏感信息"""
    errors = []
    sensitive_patterns = [
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "private_key",
        "aws_access_key_id",
        "aws_secret_access_key",
    ]

    cells = notebook.get("cells", [])
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            for pattern in sensitive_patterns:
                if pattern.lower() in source.lower():
                    errors.append(
                        f"{filepath}: Cell {i} 可能包含敏感信息 "
                        f"(检测到 '{pattern}')"
                    )

    return errors


def check_cell_outputs(notebook: dict, filepath: Path, max_output_size: int = 10000) -> list[str]:
    """检查 cell 输出是否过大"""
    errors = []
    cells = notebook.get("cells", [])

    for i, cell in enumerate(cells):
        outputs = cell.get("outputs", [])
        for j, output in enumerate(outputs):
            output_type = output.get("output_type", "")

            # 检查文本输出
            if output_type in ("stream", "execute_result"):
                text = output.get("text", [])
                if isinstance(text, list):
                    text = "".join(text)
                if len(text) > max_output_size:
                    errors.append(
                        f"{filepath}: Cell {i} 的输出 {j} 过大 "
                        f"({len(text)} 字符 > {max_output_size})"
                    )

            # 检查错误输出
            if output_type == "error":
                traceback = output.get("traceback", [])
                if len(traceback) > 50:  # 限制 traceback 行数
                    errors.append(
                        f"{filepath}: Cell {i} 的错误输出过长 "
                        f"({len(traceback)} 行)"
                    )

    return errors


def check_absolute_paths(notebook: dict, filepath: Path) -> list[str]:
    """检查是否包含绝对路径"""
    errors = []
    cells = notebook.get("cells", [])

    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))

            # 检查常见的绝对路径模式
            absolute_patterns = [
                "/home/",
                "/Users/",
                "C:\\\\",
                "D:\\\\",
                "file:///home/",
                "file:///Users/",
            ]

            for pattern in absolute_patterns:
                if pattern in source:
                    errors.append(
                        f"{filepath}: Cell {i} 包含绝对路径 '{pattern}'，"
                        f"建议使用相对路径"
                    )

    return errors


def check_notebook(filepath: Path) -> list[str]:
    """检查单个 notebook 文件"""
    errors = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except json.JSONDecodeError as e:
        return [f"{filepath}: JSON 解析错误 - {e}"]
    except Exception as e:
        return [f"{filepath}: 读取错误 - {e}"]

    # 检查 notebook 格式版本
    if notebook.get("nbformat", 0) < 4:
        errors.append(f"{filepath}: 建议使用 nbformat 4 或更高版本")

    # 运行各项检查
    errors.extend(check_kernel_name(notebook, filepath))
    errors.extend(check_sensitive_info(notebook, filepath))
    errors.extend(check_cell_outputs(notebook, filepath))
    errors.extend(check_absolute_paths(notebook, filepath))

    return errors


def main() -> int:
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python check_notebook.py <notebook_file>...")
        return 1

    all_errors = []

    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        if not filepath.exists():
            all_errors.append(f"{filepath}: 文件不存在")
            continue

        if not filepath.suffix == ".ipynb":
            continue

        errors = check_notebook(filepath)
        all_errors.extend(errors)

    if all_errors:
        print("发现以下问题:")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    print("所有 notebook 检查通过！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
