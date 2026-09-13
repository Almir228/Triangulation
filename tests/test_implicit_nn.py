"""Run with python3 -m pytest tests/test_implicit_nn.py (torch optional)."""
import json
from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from minsurf_nn.data import SurfaceDataset, read_manifest, resample_boundary, split_records
from minsurf_nn.losses import surface_loss
from minsurf_nn.model import ConditionalSDF, ModelConfig


def test_encoder_permutation_and_finite_training_step():
    torch.manual_seed(12)
    torch.set_num_threads(1)
    model = ConditionalSDF(ModelConfig(latent_dim=8, hidden_dim=16, layers=2, frequencies=2))
    boundary = torch.randn(2, 8, 3)
    query = torch.randn(2, 12, 3) * 0.05
    expected = model(boundary, query)
    torch.testing.assert_close(expected, model(boundary[:, torch.randperm(8)], query))
    assert expected.shape == (2, 12)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.decoder[-1].weight.detach().clone()
    loss, metrics = surface_loss(model, {"boundary": boundary, "query": query, "sdf": query[..., 2]}, eikonal_samples=8)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.decoder[-1].weight)
    assert metrics["eikonal"] >= 0
    assert metrics["minimality"] >= 0


def test_plane_has_zero_mean_curvature():
    class Plane(torch.nn.Module):
        def encode(self, boundary):
            return boundary.mean(dim=1)

        def decode(self, context, query):
            return query[..., 2] + context[:, None, 2] * 0.0

    from minsurf_nn.losses import _mean_curvature
    plane = Plane()
    boundary = torch.zeros(1, 4, 3)
    query = torch.randn(1, 7, 3)
    curvature = _mean_curvature(plane, plane.encode(boundary), query, training=True)
    torch.testing.assert_close(curvature, torch.zeros_like(curvature), atol=1e-7, rtol=0)


def test_manifest_loader_and_split(tmp_path):
    contour = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    records = []
    for i in range(4):
        filename = f"sample{i}.npz"
        query = np.random.default_rng(i).normal(size=(32, 3)).astype(np.float32)
        np.savez(tmp_path / filename, boundary_points=contour, query_points=query, signed_distance=query[:, 2])
        records.append({"path": filename, "group_id": str(i // 2)})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    loaded = read_manifest(manifest)
    train, val = split_records(loaded, seed=42)
    assert {r["group_id"] for r in train}.isdisjoint(r["group_id"] for r in val)
    dataset = SurfaceDataset(train, boundary_count=10, query_count=12)
    first = dataset[0]
    assert first["boundary"].shape == (10, 3)
    assert first["query"].shape == (12, 3)
    torch.testing.assert_close(first["sdf"], first["query"][:, 2])
    torch.testing.assert_close(first["query"], dataset[0]["query"])
    dataset.set_epoch(1)
    assert not torch.equal(first["query"], dataset[0]["query"])
    bad = [dict(r, split="train" if i % 2 == 0 else "val") for i, r in enumerate(loaded)]
    with pytest.raises(ValueError, match="leaks"):
        split_records(bad)


def test_closed_contour_resampling():
    contour = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    samples = resample_boundary(contour, 8)
    np.testing.assert_allclose(samples[::2], contour[:-1], atol=1e-6)
    with pytest.raises(ValueError):
        resample_boundary(np.zeros((4, 3)), 8)


def test_explicit_holdout_is_never_used_for_training():
    records = [{"path": "a", "split": "train"}, {"path": "b", "split": "val"},
               {"path": "c", "split": "test"}]
    train, val = split_records(records)
    assert [r["path"] for r in train] == ["a"]
    assert [r["path"] for r in val] == ["b"]
