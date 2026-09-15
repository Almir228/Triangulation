from math import comb
from pathlib import Path
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from minsurf_nn.spectral import evaluate_polynomial  # noqa: E402
from minsurf_nn.spectral_losses import (  # noqa: E402
    boundary_distance_loss,
    boundary_zero_loss,
    eikonal_loss,
    minimal_surface_loss,
    normalized_coefficient_loss,
    surface_geometry_losses,
    weighted_coefficient_loss,
)
from minsurf_nn.spectral_operator import (  # noqa: E402
    ContourToChebyshev,
    SpectralOperatorConfig,
    evaluate_total_degree_chebyshev,
    evaluate_total_degree_chebyshev_geometry,
)


def test_variable_point_counts_and_cyclic_shift_invariance():
    torch.manual_seed(8)
    model = ContourToChebyshev(SpectralOperatorConfig(
        degree=3, point_width=16, latent_dim=24, hidden_dim=32))
    assert model(torch.randn(2, 64, 3)).shape == (2, comb(6, 3))
    assert model(torch.randn(2, 128, 3)).shape == (2, comb(6, 3))
    contour = torch.randn(2, 252, 3)
    torch.testing.assert_close(model(contour), model(torch.roll(contour, 37, dims=1)),
                               rtol=1e-5, atol=1e-6)


def test_padding_mask_matches_unpadded_contour():
    torch.manual_seed(9)
    model = ContourToChebyshev(SpectralOperatorConfig(
        degree=2, point_width=12, latent_dim=16, hidden_dim=20))
    contour = torch.randn(1, 17, 3)
    padded = torch.cat((contour, torch.randn(1, 8, 3)), dim=1)
    mask = torch.zeros(1, 25, dtype=torch.bool)
    mask[:, :17] = True
    torch.testing.assert_close(model(contour), model(padded, mask), atol=1e-7, rtol=1e-6)


def test_torch_total_degree_evaluation_matches_numpy_and_backpropagates():
    rng = np.random.default_rng(31)
    model = ContourToChebyshev(SpectralOperatorConfig(
        degree=3, point_width=12, latent_dim=16, hidden_dim=20))
    coefficients = rng.normal(size=(2, 20)).astype(np.float32)
    points = rng.uniform(-0.8, 0.8, size=(2, 11, 3)).astype(np.float32)
    actual = evaluate_total_degree_chebyshev(
        torch.from_numpy(coefficients), torch.from_numpy(points), model.indices)
    expected = np.stack([
        evaluate_polynomial(points[i], coefficients[i], model.indices.numpy())
        for i in range(2)
    ])
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-5, atol=2e-5)

    contour = torch.randn(2, 64, 3)
    prediction = model(contour)
    target = torch.randn_like(prediction)
    loss, metrics = weighted_coefficient_loss(
        prediction, target, model.l2_weights, target_power=0.2)
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.grad is not None)
    assert float(metrics["relative_l2"]) >= 0.0


def test_future_geometry_losses_are_finite():
    model = ContourToChebyshev(SpectralOperatorConfig(
        degree=2, point_width=12, latent_dim=16, hidden_dim=20))
    boundary = torch.randn(2, 32, 3) * 0.2
    coefficients = model(boundary)
    assert torch.isfinite(boundary_zero_loss(coefficients, boundary, model.indices))
    points = torch.randn(2, 9, 3) * 0.2
    loss = (eikonal_loss(coefficients, points, model.indices) +
            0.001 * minimal_surface_loss(
                coefficients, points, model.indices, band=0.2))
    assert torch.isfinite(loss)


def test_analytic_gradient_and_hessian_match_point_autograd():
    torch.manual_seed(81)
    model = ContourToChebyshev(SpectralOperatorConfig(
        degree=3, point_width=12, latent_dim=16, hidden_dim=20))
    coefficients = torch.randn(2, 20)
    points = (torch.rand(2, 7, 3) * 1.6 - 0.8).requires_grad_(True)
    values, gradient, hessian = evaluate_total_degree_chebyshev_geometry(
        coefficients, points, model.indices)
    reference_values = evaluate_total_degree_chebyshev(
        coefficients, points, model.indices)
    reference_gradient = torch.autograd.grad(
        reference_values.sum(), points, create_graph=True)[0]
    reference_hessian = torch.stack([
        torch.autograd.grad(reference_gradient[..., axis].sum(), points,
                            retain_graph=True)[0]
        for axis in range(3)
    ], dim=-2)
    torch.testing.assert_close(values, reference_values, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(gradient, reference_gradient, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(hessian, reference_hessian, rtol=2e-5, atol=2e-5)


def test_hybrid_losses_have_expected_plane_behavior():
    model = ContourToChebyshev(SpectralOperatorConfig(
        degree=2, point_width=12, latent_dim=16, hidden_dim=20))
    coefficients = torch.zeros(2, 10)
    z_index = torch.nonzero((model.indices == torch.tensor([0, 0, 1])).all(dim=1),
                            as_tuple=False).item()
    coefficients[:, z_index] = 1.0
    points = torch.rand(2, 13, 3) * 1.6 - 0.8
    geometry = surface_geometry_losses(
        coefficients, points, model.indices, graph_margin=0.05)
    assert float(geometry["curvature"]) < 1e-10
    assert float(geometry["eikonal"]) < 1e-10
    assert float(geometry["graph"]) == 0.0

    boundary = points.clone()
    boundary[..., 2] = 0.2
    base = boundary_distance_loss(coefficients, boundary, model.indices)
    scaled = boundary_distance_loss(10.0 * coefficients, boundary, model.indices)
    torch.testing.assert_close(base, scaled, rtol=2e-5, atol=2e-7)

    target = torch.randn(2, 10)
    mean = torch.randn(10)
    scale = torch.rand(10) + 0.1
    normalized_target = (target - mean) / scale
    loss, metrics = normalized_coefficient_loss(
        normalized_target, target, mean, scale)
    assert float(loss) < 1e-12
    assert float(metrics["normalized_rmse"]) < 1e-6
