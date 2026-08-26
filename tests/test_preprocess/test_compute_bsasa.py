from inm.preprocess.compute_bsasa import compute_bsasa_value


def test_compute_bsasa_value():
    sasa1 = 1000.0
    sasa2 = 800.0
    sasa12 = 1500.0
    bsasa = compute_bsasa_value(sasa1, sasa2, sasa12)
    assert bsasa == 300.0  # 1000 + 800 - 1500


def test_compute_bsasa_value_no_contact():
    """When chains don't contact, BSASA should be ~0."""
    bsasa = compute_bsasa_value(1000.0, 800.0, 1800.0)
    assert bsasa == 0.0
