import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "inm.preprocess", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "parse-apms" in result.stdout or "parse_apms" in result.stdout
    assert "run-all" in result.stdout or "run_all" in result.stdout
