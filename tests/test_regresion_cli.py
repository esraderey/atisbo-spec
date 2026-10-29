import pickle
import subprocess
import sys
from pathlib import Path


def test_cli_trazas_y_render(tmp_path):
    root = Path(__file__).resolve().parents[1]
    pkl = tmp_path / "trazas.pkl"
    mp4 = tmp_path / "ep0.mp4"
    r1 = subprocess.run(
        [sys.executable, "-m", "atisbo.env.trazas", "--seed", "0", "--n", "10", "--p", "0.02", "--out", str(pkl)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run(
        [sys.executable, "-m", "atisbo.env.render", "--episodio", str(pkl), "--idx", "0", "--out", str(mp4)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 0, r2.stderr
    with open(pkl, "rb") as f:
        eps = pickle.load(f)
    assert len(eps) == 10
    assert mp4.stat().st_size > 1024
