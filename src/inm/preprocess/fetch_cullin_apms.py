"""Fetch the cullin AP-MS dataset (mmc2.xlsx) from Cell Host & Microbe 2019.

The publisher URL is behind Cloudflare bot-protection, so automated download
is blocked. If fetch() raises FileNotFoundError, download the file manually:

  URL: https://www.cell.com/cms/10.1016/j.chom.2019.05.008/attachment/
       6ecde5e4-3eb2-4e3b-be96-cce9fa772892/mmc2.xlsx
  DOI: https://doi.org/10.1016/j.chom.2019.05.008  (Table S2 / mmc2.xlsx)

Place the file at the path shown in the error message, then re-run.
SHA1: f237428b7f24409369864e3510106634415fc056
"""

import hashlib
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

CULLIN_APMS_URL = (
    "https://www.cell.com/cms/10.1016/j.chom.2019.05.008"
    "/attachment/6ecde5e4-3eb2-4e3b-be96-cce9fa772892/mmc2.xlsx"
)
CULLIN_APMS_FILENAME = "1-s2.0-S1931312819302537-mmc2.xlsx"
CULLIN_APMS_SHA1 = "f237428b7f24409369864e3510106634415fc056"

_MANUAL_INSTRUCTIONS = (
    "The publisher blocks automated downloads (Cloudflare). "
    "Download manually from:\n"
    "  https://doi.org/10.1016/j.chom.2019.05.008  (Table S2 / mmc2.xlsx)\n"
    "and place it at: {dest}"
)


def verify(path) -> bool:
    """Return True if path exists and its SHA1 matches the expected checksum."""
    path = Path(path)
    return path.exists() and _sha1(path) == CULLIN_APMS_SHA1


def fetch(output_dir) -> Path:
    """Download mmc2.xlsx to output_dir and verify its SHA1 checksum.

    Raises FileNotFoundError with manual-download instructions if the
    publisher blocks the request (HTTP 403 / Cloudflare challenge).
    Returns the path to the verified file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / CULLIN_APMS_FILENAME

    if dest.exists():
        if _sha1(dest) == CULLIN_APMS_SHA1:
            logger.info("cullin AP-MS data already present and verified: %s", dest)
            return dest
        logger.warning("existing file has wrong checksum, re-downloading")
        dest.unlink()

    logger.info("downloading cullin AP-MS data from %s", CULLIN_APMS_URL)
    req = urllib.request.Request(
        CULLIN_APMS_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; fetch_cullin_apms/1.0)"},
    )
    try:
        with urllib.request.urlopen(req) as response, open(dest, "wb") as out:
            out.write(response.read())
    except urllib.error.HTTPError as e:
        raise FileNotFoundError(_MANUAL_INSTRUCTIONS.format(dest=dest)) from e

    actual = _sha1(dest)
    if actual != CULLIN_APMS_SHA1:
        dest.unlink()
        raise ValueError(
            f"SHA1 mismatch for {dest.name}: expected {CULLIN_APMS_SHA1}, got {actual}"
        )

    logger.info("downloaded and verified %s", dest)
    return dest


def run(cfg: Dict[str, Any], output_dir) -> None:
    """Fetch the cullin AP-MS xlsx; cfg is unused (URL is fixed)."""
    fetch(output_dir)


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
