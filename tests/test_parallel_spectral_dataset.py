"""Integration checks for resumable parallel dataset-wide Chebyshev fitting."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plane_mesh():
    vertices = np.array([
        [-0.75, -0.75, 0.0], [0.0, -0.75, 0.0], [0.75, -0.75, 0.0],
        [-0.75, 0.0, 0.0], [0.0, 0.0, 0.0], [0.75, 0.0, 0.0],
        [-0.75, 0.75, 0.0], [0.0, 0.75, 0.0], [0.75, 0.75, 0.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7],
    ], dtype=np.int32)
    return vertices, faces


class ParallelSpectralDatasetTests(unittest.TestCase):
    def test_parallel_fitting_writes_resumes_and_recovers(self):
        with tempfile.TemporaryDirectory(prefix="parallel_spectral_test_") as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "coefficients"
            source.mkdir()
            vertices, faces = plane_mesh()
            source_records = []
            for index in range(4):
                path = source / f"sample_{index:06d}.npz"
                shifted = vertices.copy()
                shifted[:, 2] = 0.02 * index
                np.savez(path, surface_vertices=shifted, faces=faces)
                source_records.append({
                    "index": index,
                    "sample_id": f"sample_{index:06d}",
                    "path": path.name,
                    "seed": 100 + index,
                    "split": "val" if index == 3 else "train",
                })
            source_manifest = source / "manifest.jsonl"
            source_manifest.write_text("".join(
                json.dumps(record) + "\n" for record in source_records), encoding="utf-8")

            command = [
                sys.executable, str(ROOT / "scripts" / "fit_chebyshev_dataset_parallel.py"),
                str(source_manifest), "--output-dir", str(output),
                "--degree", "3", "--surface-samples", "240",
                "--boundary-samples", "80", "--workers", "2",
                "--shard-size", "2", "--progress-every", "1",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            manifest_path = output / "manifest.jsonl"
            complete_manifest = manifest_path.read_text(encoding="utf-8")
            records = [json.loads(line) for line in complete_manifest.splitlines()]
            self.assertEqual([record["index"] for record in records], list(range(4)))
            self.assertEqual(len(list(output.glob("shard-*/*.npz"))), 4)
            for record in records:
                with np.load(output / record["coefficient_path"], allow_pickle=False) as fit:
                    self.assertEqual(fit["coefficients"].shape, (20,))
                    self.assertEqual(fit["indices"].shape, (20, 3))
                    self.assertEqual(fit["coefficient_energy"].shape, (4,))
                    metadata = json.loads(str(fit["metadata"]))
                    self.assertEqual(metadata["source_index"], record["index"])
                    self.assertEqual(metadata["diagnostics"]["rank"], 20)

            manifest_path.write_text(
                "\n".join(complete_manifest.splitlines()[:-1]) + "\n", encoding="utf-8")
            recovered = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("recovered 1", recovered.stdout)
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), complete_manifest)
            resumed = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("already complete", resumed.stdout)
            summary = json.loads((output / "fitting_summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["completed_samples"], 4)
            self.assertEqual(summary["full_rank_samples"], 4)

            mismatch = subprocess.run(command + ["--degree", "4"],
                                      capture_output=True, text=True)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("different fitting configuration", mismatch.stderr)


if __name__ == "__main__":
    unittest.main()
