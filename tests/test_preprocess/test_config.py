from pathlib import Path
import pytest
from inm.preprocess.config import load_config

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_load_config_returns_required_keys():
    cfg = load_config(FIXTURE_DIR / "test_config.yaml")
    assert cfg["dataset_name"] == "test"
    assert cfg["sheet_nums"] == [0]
    assert cfg["filter"]["max_evalue"] == 1e-7


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))
