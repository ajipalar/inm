"""Integration tests for examples/cullin_example/scripts/preprocess.py."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from tests.test_preprocess.conftest import FIXTURE_DIR


def test_cullin_example_preprocess_script_success(ensure_dummy_apms):
    """Test cullin_example preprocess.py script with valid config."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        config = {
            "data": {"xlsx_path": str(FIXTURE_DIR / "dummy_apms.xlsx")},
            "preprocess": {
                "sheet_nums": [0],
                "bait_col": "Bait",
                "prey_col": "Prey",
                "ms_score_col": "SaintScore",
                "spec_col": "Spec",
                "ctrl_col": "ctrlCounts",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable,
                "examples/cullin_example/scripts/preprocess.py",
                str(config_path),
                str(output_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        spec_table = pd.read_csv(
            output_dir / "spec_table.tsv", sep="\t", header=None, index_col=0
        )
        assert isinstance(spec_table, pd.DataFrame)
        assert len(spec_table) > 0

        composites = pd.read_csv(output_dir / "composite_table.tsv", sep="\t")
        assert isinstance(composites, pd.DataFrame)
        assert len(composites) > 0


def test_cullin_example_preprocess_missing_config():
    """Test that script exits non-zero when config file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        result = subprocess.run(
            [
                sys.executable,
                "examples/cullin_example/scripts/preprocess.py",
                str(tmp_path / "nonexistent.yaml"),
                str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0


def test_cullin_example_preprocess_invalid_yaml():
    """Test that script exits non-zero when config YAML is malformed."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        config_path = tmp_path / "bad.yaml"
        config_path.write_text("invalid: yaml: content: [")

        result = subprocess.run(
            [
                sys.executable,
                "examples/cullin_example/scripts/preprocess.py",
                str(config_path),
                str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0


def test_cullin_example_preprocess_missing_config_keys():
    """Test that script exits non-zero when required config keys are missing."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        config = {"data": {"xlsx_path": "file.xlsx"}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = subprocess.run(
            [
                sys.executable,
                "examples/cullin_example/scripts/preprocess.py",
                str(config_path),
                str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0


def test_cullin_example_preprocess_missing_xlsx_key():
    """Test that script exits non-zero when xlsx_path is missing from config."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        config = {
            "data": {},
            "preprocess": {
                "sheet_nums": [0],
                "bait_col": "Bait",
                "prey_col": "Prey",
                "ms_score_col": "SaintScore",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = subprocess.run(
            [
                sys.executable,
                "examples/cullin_example/scripts/preprocess.py",
                str(config_path),
                str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
