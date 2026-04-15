"""基础单元测试 - 验证项目结构和基本功能"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProjectStructure:
    """测试项目结构"""

    def test_project_root_exists(self):
        """测试项目根目录存在"""
        root = Path(__file__).parent.parent
        assert root.exists()
        assert (root / "mingpt").exists()
        assert (root / "pyproject.toml").exists()

    def test_mingpt_package_exists(self):
        """测试 mingpt 包存在"""
        mingpt_dir = Path(__file__).parent.parent / "mingpt"
        assert mingpt_dir.exists()
        assert (mingpt_dir / "__init__.py").exists()
        assert (mingpt_dir / "model.py").exists()
        assert (mingpt_dir / "bpe.py").exists()
        assert (mingpt_dir / "trainer.py").exists()

    def test_config_files_exist(self):
        """测试配置文件存在"""
        root = Path(__file__).parent.parent
        assert (root / "pyproject.toml").exists()
        assert (root / ".pre-commit-config.yaml").exists()
        assert (root / "MANIFEST.in").exists()

    def test_github_workflows_exist(self):
        """测试 GitHub Actions 工作流存在"""
        workflows_dir = Path(__file__).parent.parent / ".github" / "workflows"
        assert workflows_dir.exists()
        assert (workflows_dir / "ci.yml").exists()
        assert (workflows_dir / "release.yml").exists()


class TestImports:
    """测试导入功能"""

    def test_import_mingpt(self):
        """测试导入 mingpt 包"""
        import mingpt

        assert mingpt is not None

    def test_import_model(self):
        """测试导入 model 模块"""
        from mingpt.model import GPT

        assert GPT is not None

    def test_import_trainer(self):
        """测试导入 trainer 模块"""
        from mingpt.trainer import Trainer

        assert Trainer is not None

    def test_import_bpe(self):
        """测试导入 bpe 模块"""
        from mingpt.bpe import BPETokenizer

        assert BPETokenizer is not None


class TestModelBasic:
    """测试模型基本功能"""

    def test_gpt_config(self):
        """测试 GPT 配置"""
        from mingpt.model import GPT

        config = GPT.get_default_config()
        assert config is not None

    def test_trainer_config(self):
        """测试 Trainer 配置"""
        from mingpt.trainer import Trainer

        config = Trainer.get_default_config()
        assert config is not None


class TestPyprojectToml:
    """测试 pyproject.toml 配置"""

    def test_pyproject_parsable(self):
        """测试 pyproject.toml 可解析"""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        root = Path(__file__).parent.parent
        with open(root / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        assert "project" in config
        assert config["project"]["name"] == "mingpt"

    def test_build_system_config(self):
        """测试构建系统配置"""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        root = Path(__file__).parent.parent
        with open(root / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        assert "build-system" in config
        assert "setuptools.build_meta" in config["build-system"]["build-backend"]


class TestPrecommitConfig:
    """测试 pre-commit 配置"""

    def test_precommit_config_parsable(self):
        """测试 .pre-commit-config.yaml 可解析"""
        import yaml

        root = Path(__file__).parent.parent
        with open(root / ".pre-commit-config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "repos" in config
        assert len(config["repos"]) > 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
