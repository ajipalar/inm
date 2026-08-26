"""Fetch the Predictomes pair-score dump (Schmidt et al., Mol Cell 2025).

Paper: Schmidt et al. (2025). Mol Cell. https://doi.org/10.1016/j.molcel.2025.01.034
Database: https://predictomes.org (bulk download listed under /downloads)

Downloads the gzip-compressed CSV of AlphaFold-Multimer-modeled human PPI
pairs and decompresses it to the path referenced by
config["predictomes"]["raw_path"] (default: data/predictome/<filename>).
"""

import gzip
import hashlib
import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

PREDICTOME_URL = (
    "https://predictomes-hsbps-dataset.s3.us-east-1.amazonaws.com"
    "/20251110_hs_predictome_pair_scores.csv.gz"
)
PREDICTOME_FILENAME = "20251110_hs_predictome_pair_scores.csv"
PREDICTOME_SHA1 = "3a52cd7aecbc647f08bb0cd10255c95d761fcb96"


def verify(path) -> bool:
    """Return True if path exists and its SHA1 matches the expected checksum."""
    path = Path(path)
    return path.exists() and _sha1(path) == PREDICTOME_SHA1


def fetch(dest) -> Path:
    """Download and decompress the Predictomes pair-score CSV to dest, verifying its SHA1.

    Returns the path to the verified file. Raises ValueError on checksum mismatch.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if _sha1(dest) == PREDICTOME_SHA1:
            logger.info("Predictomes data already present and verified: %s", dest)
            return dest
        logger.warning("existing file has wrong checksum, re-downloading")
        dest.unlink()

    logger.info("downloading Predictomes pair scores from %s", PREDICTOME_URL)
    req = urllib.request.Request(
        PREDICTOME_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; fetch_predictome/1.0)"},
    )
    gz_path = dest.with_suffix(dest.suffix + ".gz")
    try:
        with urllib.request.urlopen(req) as response, open(gz_path, "wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as e:
        raise FileNotFoundError(
            f"Could not download {PREDICTOME_URL} ({e}). "
            f"Download manually from https://predictomes.org/downloads "
            f"and decompress to {dest}"
        ) from e

    with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()

    actual = _sha1(dest)
    if actual != PREDICTOME_SHA1:
        dest.unlink()
        raise ValueError(
            f"SHA1 mismatch for {dest.name}: expected {PREDICTOME_SHA1}, got {actual}"
        )

    logger.info("downloaded and verified %s", dest)
    return dest


def run(cfg: Dict[str, Any], config_dir) -> None:
    """Fetch the Predictomes CSV to the path in cfg["predictomes"]["raw_path"].

    raw_path is resolved relative to config_dir (the example's own directory),
    matching how Snakemake resolves it.
    """
    dest = Path(config_dir) / cfg["predictomes"]["raw_path"]
    fetch(dest)


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
