import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_dataset import prepare_canonical_contour  # noqa: E402
from minsurf_nn.data import (AugmentedChebyshevDataset,
                             ChebyshevRotationPackDataset,
                             read_manifest)  # noqa: E402
from minsurf_nn.rotations import rotate_xy_points  # noqa: E402
from minsurf_nn.spectral import evaluate_polynomial, total_degree_indices  # noqa: E402


def _make_input(root: Path, count: int = 3) -> Path:
    source = root / "source"
    coefficients_dir = root / "coefficients"
    source.mkdir()
    coefficients_dir.mkdir()
    indices = total_degree_indices(3)
    rng = np.random.default_rng(71)
    records = []
    for index in range(count):
        boundary = np.asarray([
            [-0.6, -0.5, 0.02 * index], [0.6, -0.5, -0.01 * index],
            [0.6, 0.5, 0.01 * index], [-0.6, 0.5, -0.02 * index],
        ])
        surface = np.vstack((boundary, [[0.0, 0.0, 0.0]]))
        source_path = source / f"sample_{index:06d}.npz"
        coefficient_path = coefficients_dir / f"sample_{index:06d}_chebyshev.npz"
        values = rng.normal(scale=0.1, size=len(indices))
        np.savez_compressed(source_path, boundary_points=boundary,
                            surface_vertices=surface)
        np.savez_compressed(coefficient_path, coefficients=values,
                            indices=indices, degree=np.asarray(3))
        records.append({
            "index": index,
            "sample_id": f"sample_{index:06d}",
            "group_id": f"sample_{index:06d}",
            "split": "val" if index == count - 1 else "train",
            "source_path": f"../source/{source_path.name}",
            "coefficient_path": coefficient_path.name,
            "degree": 3,
        })
    manifest = coefficients_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records),
                        encoding="utf-8")
    return manifest


def test_parallel_rotation_augmentation_writes_resumes_and_loads(tmp_path):
    manifest = _make_input(tmp_path)
    output = tmp_path / "augmented"
    command = [
        sys.executable, str(ROOT / "scripts/augment_chebyshev_dataset_parallel.py"),
        str(manifest), "--output-dir", str(output), "--copies", "4",
        "--workers", "2", "--shard-size", "2", "--progress-every", "1",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    records = [json.loads(line) for line in
               (output / "manifest.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert all(record["copies"] == 4 for record in records)
    assert all(record["group_id"] == record["sample_id"] for record in records)

    first_path = output / records[0]["path"]
    with np.load(first_path, allow_pickle=False) as pack:
        assert pack["boundary_points"].shape == (4, 4, 3)
        assert pack["coefficients"].shape == (4, 20)
        assert pack["coefficient_energy"].shape == (4, 4)
        assert pack["angles"].shape == (4,)
        assert float(pack["angles"][0]) == 0.0
        np.testing.assert_allclose(pack["boundary_points"][0],
                                   np.load(tmp_path / "source/sample_000000.npz")["boundary_points"])

        source_point = np.asarray([[0.17, -0.23, 0.09]])
        original = np.load(tmp_path / "coefficients/sample_000000_chebyshev.npz")
        expected = evaluate_polynomial(source_point, original["coefficients"],
                                       original["indices"])
        for copy, angle in enumerate(pack["angles"]):
            actual = evaluate_polynomial(
                rotate_xy_points(source_point, float(angle)),
                pack["coefficients"][copy], pack["indices"])
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    loaded_records = read_manifest(output / "manifest.jsonl")
    dataset = AugmentedChebyshevDataset(loaded_records, boundary_count=8)
    assert len(dataset) == 12
    assert dataset[0]["boundary"].shape == (8, 3)
    assert dataset[0]["coefficients"].shape == (20,)
    assert int(dataset[-1]["copy_index"]) == 3

    pack_dataset = ChebyshevRotationPackDataset(loaded_records, boundary_count=8)
    assert len(pack_dataset) == 3
    assert pack_dataset[0]["boundary"].shape == (4, 8, 3)
    assert pack_dataset[0]["coefficients"].shape == (4, 20)

    resumed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stderr
    assert "already complete" in resumed.stdout
    summary = json.loads((output / "augmentation_summary.json").read_text())
    assert summary["complete"]
    assert summary["augmented_samples"] == 12
    assert summary["maximum_rotation_identity_error"] < 1e-11

    mismatch = subprocess.run(command + ["--copies", "5"],
                              check=False, capture_output=True, text=True)
    assert mismatch.returncode != 0
    assert "different augmentation configuration" in mismatch.stderr


def test_augmentation_can_regenerate_contours_without_source_npz(tmp_path):
    manifest = _make_input(tmp_path, count=1)
    source = tmp_path / "source"
    generation = {
        "samples": 1,
        "seed": 1234,
        "boundary_size": 64,
        "raw_contour_size": 128,
        "fourier_modes": 5,
        "fourier_decay": 2.5,
        "xy_variation": 0.22,
        "nonplanarity": 0.32,
        "min_separation": 0.025,
        "max_curvature": 25.0,
    }
    (source / "generation_config.json").write_text(
        json.dumps(generation), encoding="utf-8")
    (manifest.parent / "fitting_config.json").write_text(json.dumps({
        "source_manifest": str(source / "manifest.jsonl"),
    }), encoding="utf-8")
    for path in source.glob("sample_*.npz"):
        path.unlink()

    output = tmp_path / "regenerated"
    completed = subprocess.run([
        sys.executable, str(ROOT / "scripts/augment_chebyshev_dataset_parallel.py"),
        str(manifest), "--output-dir", str(output), "--copies", "2", "--workers", "1",
    ], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "contour_source=regenerate" in completed.stdout
    record = json.loads((output / "manifest.jsonl").read_text())
    expected, *_ = prepare_canonical_contour(
        index=0, base_seed=1234, boundary_count=64, raw_contour_count=128,
        fourier_modes=5, fourier_decay=2.5, xy_variation=0.22,
        nonplanarity=0.32, min_separation=0.025, max_curvature=25.0)
    with np.load(output / record["path"], allow_pickle=False) as pack:
        np.testing.assert_array_equal(pack["boundary_points"][0], expected)
