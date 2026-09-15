import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_viewer_renders_a_manifest_sample(tmp_path):
    source_dir = tmp_path / "source"
    coefficient_dir = tmp_path / "coefficients"
    source_dir.mkdir()
    coefficient_dir.mkdir()

    vertices = np.asarray([
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    boundary = vertices.copy()
    np.savez_compressed(
        source_dir / "sample.npz",
        surface_vertices=vertices,
        faces=faces,
        dense_boundary_points=boundary,
    )

    # total_degree_indices(1) = (0,0,0), (0,0,1), (0,1,0), (1,0,0).
    indices = np.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
                         dtype=np.int16)
    np.savez_compressed(
        coefficient_dir / "sample_chebyshev.npz",
        coefficients=np.asarray([0.0, 1.0, 0.0, 0.0]),
        indices=indices,
        degree=np.asarray(1),
        coefficient_energy=np.asarray([0.0, 1.0]),
    )
    manifest = coefficient_dir / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "index": 17,
        "sample_id": "plane",
        "split": "test",
        "source_path": "../source/sample.npz",
        "coefficient_path": "sample_chebyshev.npz",
        "diagnostics": {"condition_number": 1.0},
    }) + "\n", encoding="utf-8")

    image = tmp_path / "view.png"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/view_chebyshev_fit.py"),
         str(manifest), "--index", "17", "--no-show", "--save", str(image)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"index": 17' in completed.stdout
    assert '"vertical_rmse": 0.0' in completed.stdout
    assert image.stat().st_size > 10_000
