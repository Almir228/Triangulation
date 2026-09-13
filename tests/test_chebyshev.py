"""Tests for direct contour-to-formula prediction (torch optional)."""

from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from minsurf_nn.chebyshev import (ChebyshevConfig, ConditionalChebyshev,
                                  chebyshev_basis, coefficient_regularization,
                                  evaluate_chebyshev)
from minsurf_nn.losses import surface_loss
from minsurf_nn.data import SurfaceDataset
from scripts.export_chebyshev import render_formula


def test_basis_and_known_polynomial():
    values = torch.tensor([0.0, 0.5])
    basis = chebyshev_basis(values, 3)
    expected = torch.stack((torch.ones_like(values), values,
                            2 * values.square() - 1,
                            4 * values.pow(3) - 3 * values), dim=-1)
    torch.testing.assert_close(basis, expected)
    coefficients = torch.zeros(1, 3, 3, 3)
    coefficients[0, 1, 0, 0] = 2.0
    coefficients[0, 0, 1, 0] = -1.0
    query = torch.tensor([[[0.25, -0.5, 0.1]]])
    torch.testing.assert_close(evaluate_chebyshev(coefficients, query, 1.0),
                               torch.tensor([[1.0]]))


def test_direct_coefficients_area_and_training_gradient():
    torch.manual_seed(7)
    model = ConditionalChebyshev(ChebyshevConfig(
        degree=2, extent=1.25, latent_dim=8, hidden_dim=12))
    boundary = torch.randn(2, 10, 3)
    query = torch.randn(2, 16, 3) * 0.4
    context = model.encode(boundary)
    coefficients = model.coefficients_from_context(context)
    area = model.area_from_context(context)
    assert coefficients.shape == (2, 3, 3, 3)
    assert torch.all(area > 0)
    torch.testing.assert_close(model.encode(boundary[:, torch.randperm(10)]), context)
    loss, _ = surface_loss(model, {"boundary": boundary, "query": query,
                                   "sdf": query[..., 2]},
                           eikonal_samples=8, minimal_weight=0)
    total = loss + 1e-4 * coefficient_regularization(coefficients)
    total.backward()
    assert model.coefficient_head.weight.grad is not None
    assert torch.isfinite(model.coefficient_head.weight.grad).all()


def test_exported_formula_matches_coefficients():
    coefficients = np.zeros((3, 3, 3), dtype=np.float64)
    coefficients[0, 0, 0] = 0.2
    coefficients[1, 0, 0] = 0.4
    coefficients[0, 2, 0] = -0.3
    center = np.array([1.0, -2.0, 0.5])
    scale, extent = 2.0, 1.3
    boundary = np.array([[0, -3, 0], [2, -3, 0], [2, -1, 0], [0, -1, 0]])
    source = render_formula(coefficients, center, scale, extent, 7.5, boundary)
    namespace = {}
    exec(compile(source, "generated_formula.py", "exec"), namespace)
    point = np.array([1.4, -1.2, 0.7])
    query = torch.tensor(((point - center) / scale).reshape(1, 1, 3), dtype=torch.float64)
    expected = evaluate_chebyshev(
        torch.tensor(coefficients).unsqueeze(0), query, extent).item()
    assert namespace["F"](*point) == pytest.approx(expected, abs=1e-12)
    assert namespace["PREDICTED_AREA"] == 7.5
    assert namespace["inside_domain"](1.0, -2.0)
    assert not namespace["inside_domain"](3.0, -2.0)


def test_dataset_returns_normalized_target_area(tmp_path):
    boundary = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                        dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    query = np.zeros((4, 3), dtype=np.float32)
    path = tmp_path / "sample.npz"
    np.savez(path, boundary_points=boundary, query_points=query,
             signed_distance=np.zeros(4, dtype=np.float32),
             surface_vertices=boundary, faces=faces)
    item = SurfaceDataset([{"path": str(path)}], boundary_count=4,
                          query_count=4, return_area=True)[0]
    assert item["area"].item() == pytest.approx(1.0)
