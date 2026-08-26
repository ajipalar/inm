"""Tests for fetch_predictome — verifies SHA1 against the canonical local file."""

import shutil
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from inm.preprocess.fetch_predictome import (
    PREDICTOME_FILENAME,
    PREDICTOME_SHA1,
    _sha1,
    fetch,
    verify,
)

# Path to the canonical copy already present in the repo (not checked into git)
_CANONICAL = (
    Path(__file__).parent.parent.parent  # inm/
    / "data"
    / "predictome"
    / PREDICTOME_FILENAME
)


# ---------------------------------------------------------------------------
# SHA1 / verify
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _CANONICAL.exists(), reason="canonical predictome CSV not present"
)
def test_canonical_sha1():
    """The known-good file in data/predictome/ must match the recorded checksum."""
    assert _sha1(_CANONICAL) == PREDICTOME_SHA1


@pytest.mark.skipif(
    not _CANONICAL.exists(), reason="canonical predictome CSV not present"
)
def test_verify_canonical():
    assert verify(_CANONICAL)


def test_verify_missing(tmp_path):
    assert not verify(tmp_path / PREDICTOME_FILENAME)


def test_verify_wrong_content(tmp_path):
    bad = tmp_path / PREDICTOME_FILENAME
    bad.write_bytes(b"not the real file")
    assert not verify(bad)


# ---------------------------------------------------------------------------
# fetch() — already-present fast path (no network)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _CANONICAL.exists(), reason="canonical predictome CSV not present"
)
def test_fetch_skips_when_verified(tmp_path):
    """fetch() must return immediately if the file is already verified."""
    dest = tmp_path / PREDICTOME_FILENAME
    shutil.copy(_CANONICAL, dest)

    with patch("urllib.request.urlopen") as mock_open:
        result = fetch(dest)

    mock_open.assert_not_called()
    assert result == dest


# ---------------------------------------------------------------------------
# fetch() — bad-checksum triggers re-download attempt
# ---------------------------------------------------------------------------


def test_fetch_removes_corrupt_file_and_raises(tmp_path):
    """fetch() removes a corrupt file and raises FileNotFoundError on HTTP 403."""
    dest = tmp_path / PREDICTOME_FILENAME
    dest.write_bytes(b"corrupt")

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(None, 403, "Forbidden", {}, None),
    ):
        with pytest.raises(FileNotFoundError, match="manually"):
            fetch(dest)

    assert not dest.exists()


# ---------------------------------------------------------------------------
# fetch() — HTTP 403 on fresh download
# ---------------------------------------------------------------------------


def test_fetch_403_raises_file_not_found(tmp_path):
    dest = tmp_path / PREDICTOME_FILENAME
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(None, 403, "Forbidden", {}, None),
    ):
        with pytest.raises(FileNotFoundError, match="manually"):
            fetch(dest)
