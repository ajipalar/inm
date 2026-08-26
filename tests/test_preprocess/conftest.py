"""Generate test fixtures for preprocess tests."""

from pathlib import Path

import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# dummy_apms.xlsx schema (authoritative source; regenerate by deleting the file and running pytest)
_DUMMY_APMS_DATA = {
    "Bait": ["BAIT1", "BAIT1", "BAIT1", "BAIT1"],
    "Prey": ["P00001", "Q99999", "OLDID", "P00002"],
    "PreyGene": ["PROT_A", "IGHG1_MOUSE", "vifprotein", "PROT_B"],
    "Spec": ["2|3|1", "1|1|1", "3|2|4", "1|2|1"],
    "ctrlCounts": ["0|0|0", "0|0|0", "0|0|0", "0|0|0"],
    "SaintScore": [0.9, 0.5, 0.8, 0.7],
}


@pytest.fixture(scope="session", autouse=True)
def ensure_dummy_apms():
    """Create dummy_apms.xlsx in fixtures dir if it doesn't exist."""
    fixture_path = FIXTURE_DIR / "dummy_apms.xlsx"
    if not fixture_path.exists():
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(_DUMMY_APMS_DATA).to_excel(fixture_path, index=False)
