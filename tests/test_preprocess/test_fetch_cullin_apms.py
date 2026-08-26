"""Tests for fetch_cullin_apms — verifies SHA1 against the canonical local file."""

import shutil
from pathlib import Path
from unittest.mock import patch
import urllib.error

import pytest

from inm.preprocess.fetch_cullin_apms import (
    CULLIN_APMS_FILENAME,
    CULLIN_APMS_SHA1,
    _sha1,
    fetch,
    verify,
)

# Path to the canonical copy already present in the repo
_CANONICAL = (
    Path(__file__).parent.parent.parent  # inm/
    / "data"
    / "cullin"
    / "1-s2.0-S1931312819302537-mmc2.xlsx"
)


# ---------------------------------------------------------------------------
# SHA1 / verify
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CANONICAL.exists(), reason="canonical cullin xlsx not present")
def test_canonical_sha1():
    """The known-good file in data/cullin/ must match the recorded checksum."""
    assert _sha1(_CANONICAL) == CULLIN_APMS_SHA1


@pytest.mark.skipif(not _CANONICAL.exists(), reason="canonical cullin xlsx not present")
def test_verify_canonical():
    assert verify(_CANONICAL)


def test_verify_missing(tmp_path):
    assert not verify(tmp_path / "nonexistent.xlsx")


def test_verify_wrong_content(tmp_path):
    bad = tmp_path / CULLIN_APMS_FILENAME
    bad.write_bytes(b"not the real file")
    assert not verify(bad)


# ---------------------------------------------------------------------------
# fetch() — already-present fast path (no network)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CANONICAL.exists(), reason="canonical cullin xlsx not present")
def test_fetch_skips_when_verified(tmp_path):
    """fetch() must return immediately if the file is already verified."""
    dest = tmp_path / CULLIN_APMS_FILENAME
    shutil.copy(_CANONICAL, dest)

    with patch("urllib.request.urlopen") as mock_open:
        result = fetch(tmp_path)

    mock_open.assert_not_called()
    assert result == dest


# ---------------------------------------------------------------------------
# fetch() — bad-checksum triggers re-download attempt
# ---------------------------------------------------------------------------


def test_fetch_removes_corrupt_file_and_raises(tmp_path):
    """fetch() removes a corrupt file and raises FileNotFoundError on HTTP 403."""
    dest = tmp_path / CULLIN_APMS_FILENAME
    dest.write_bytes(b"corrupt")

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(None, 403, "Forbidden", {}, None),
    ):
        with pytest.raises(FileNotFoundError, match="manually"):
            fetch(tmp_path)

    assert not dest.exists()


# ---------------------------------------------------------------------------
# fetch() — HTTP 403 on fresh download
# ---------------------------------------------------------------------------


def test_fetch_403_raises_file_not_found(tmp_path):
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(None, 403, "Forbidden", {}, None),
    ):
        with pytest.raises(FileNotFoundError, match="manually"):
            fetch(tmp_path)
