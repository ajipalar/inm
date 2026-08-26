from pathlib import Path
from inm.preprocess.parse_hhr import parse_hhr_file

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_hhr_file_extracts_alignments():
    alns = parse_hhr_file(FIXTURE_DIR / "P00001.hhr")
    assert len(alns) >= 1
    header, meta, query, template = alns[0]
    assert isinstance(header, str)
    assert len(header) > 0
    assert isinstance(query, str)
    assert isinstance(template, str)
    assert len(query) > 0
    assert len(template) > 0


def test_parse_hhr_file_finds_multiple_alignments():
    alns = parse_hhr_file(FIXTURE_DIR / "P00001.hhr")
    assert len(alns) > 1
