import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from minsurf_nn.spectral import total_degree_indices  # noqa: E402
from minsurf_nn.spectral_operator import (ContourToChebyshev,
                                          SpectralOperatorConfig)  # noqa: E402


def _make_rotation_packs(root: Path) -> Path:
    indices = total_degree_indices(2)
    rng = np.random.default_rng(918)
    records = []
    for index in range(6):
        copies, points = 3, 16
        boundaries = rng.uniform(-0.7, 0.7, size=(copies, points, 3)).astype(np.float32)
        coefficients = rng.normal(0.0, 0.1, size=(copies, len(indices))).astype(np.float64)
        path = root / f"pack_{index:03d}.npz"
        np.savez_compressed(
            path, boundary_points=boundaries, coefficients=coefficients,
            angles=np.linspace(0.0, 1.0, copies), indices=indices,
            degree=np.asarray(2), source_index=np.asarray(index))
        records.append({
            "index": index, "sample_id": f"sample_{index}",
            "group_id": f"sample_{index}",
            "split": "train" if index < 4 else "val", "path": path.name,
            "degree": 2, "copies": copies,
        })
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records),
                        encoding="utf-8")
    return manifest


def test_supervised_trainer_smoke_and_checkpoint_is_size_independent(tmp_path):
    manifest = _make_rotation_packs(tmp_path)
    output = tmp_path / "run"
    command = [
        sys.executable, str(ROOT / "scripts/train_spectral_operator.py"),
        str(manifest), "--output-dir", str(output), "--epochs", "1",
        "--batch-packs", "2", "--boundary-count", "64", "--workers", "0",
        "--threads", "1", "--device", "cpu", "--point-width", "16",
        "--latent-dim", "20", "--hidden-dim", "24",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert (output / "last.pt").is_file()
    assert (output / "best.pt").is_file()
    assert (output / "coefficient_stats.npz").is_file()
    metrics = json.loads((output / "metrics.jsonl").read_text().strip())
    assert metrics["train_examples"] == 12
    assert metrics["validation_examples"] == 6

    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model = ContourToChebyshev(SpectralOperatorConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        for count in (64, 128, 252):
            assert model(torch.randn(2, count, 3)).shape == (2, 10)

    resumed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stderr
    assert "already complete" in resumed.stdout
