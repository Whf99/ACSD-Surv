"""Numerical structure and mixed-derivative checks on a small closed mesh."""

import pytest
import torch

from acsd_surv.geometry import SpectralGeometry, geometry_state, normal_geometry, sobolev_metric


def octahedron():
    vertices = torch.tensor([
        [1.4, 0., 0.], [-1.1, 0., 0.], [0., 1.2, 0.],
        [0., -0.9, 0.], [0., 0., 1.6], [0., 0., -0.8],
    ], dtype=torch.float64)
    faces = torch.tensor([
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ])
    return vertices, faces


def reference(vertices):
    return (vertices.new_tensor([0.2, -0.4, 0.1]),
            torch.eye(3, dtype=vertices.dtype), 4.)


def test_fem_isotropic_limit_zero_mode_and_positive_grams():
    vertices, faces = octahedron()
    model = SpectralGeometry().double()
    state = geometry_state(vertices, faces)
    torch.testing.assert_close(model.operator(vertices, faces, reference(vertices)[1]),
                               state["isotropic_operator"], rtol=1e-10, atol=1e-10)
    zero_mode = state["mass"].sqrt()
    torch.testing.assert_close(state["isotropic_operator"] @ zero_mode,
                               torch.zeros_like(zero_mode), atol=1e-10, rtol=0)
    for gram in model.response_matrices(vertices, faces, *reference(vertices)):
        assert torch.linalg.eigvalsh(gram).min() >= -1e-10
    scales = model.diffusion_scales()
    assert torch.all(scales[1:] > scales[:-1])
    assert model.features(vertices, faces, *reference(vertices)).shape == (43,)


def test_shared_rigid_coordinate_transform_preserves_features():
    vertices, faces = octahedron()
    origin, axes, length = reference(vertices)
    model = SpectralGeometry().double()
    with torch.no_grad():
        model.conductivity_net[-1].weight.normal_(std=0.2)
    angle = vertices.new_tensor(0.47)
    c, s = angle.cos(), angle.sin()
    rotation = vertices.new_tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    rotation[0, 0] = c
    rotation[0, 1] = -s
    rotation[1, 0] = s
    rotation[1, 1] = c
    translation = vertices.new_tensor([3., -2., 1.])
    raw = model.features(vertices, faces, origin, axes, length)
    transformed = model.features(vertices @ rotation.T + translation, faces,
                                 rotation @ origin + translation, rotation @ axes, length)
    torch.testing.assert_close(raw, transformed, atol=1e-9, rtol=1e-9)


def test_vertex_gradient_matches_directional_finite_difference():
    vertices, faces = octahedron()
    vertices.requires_grad_(True)
    model = SpectralGeometry().double()
    origin, axes, length = reference(vertices)
    with torch.no_grad():
        model.conductivity_net[-1].weight.normal_(std=0.1)
    torch.manual_seed(31)
    direction = torch.randn_like(vertices)
    weights = torch.linspace(0.1, 0.9, model.raw_dim, dtype=vertices.dtype)
    value = (model.features(vertices, faces, origin, axes, length) * weights).sum()
    gradient = torch.autograd.grad(value, vertices)[0]
    eps = 1e-5
    with torch.no_grad():
        plus = (model.features(vertices + eps * direction, faces, origin, axes, length) * weights).sum()
        minus = (model.features(vertices - eps * direction, faces, origin, axes, length) * weights).sum()
    torch.testing.assert_close((gradient * direction).sum(), (plus - minus) / (2 * eps),
                               rtol=3e-5, atol=3e-6)


def test_final_prediction_normal_penalty_has_mixed_parameter_gradient():
    torch.manual_seed(11)
    vertices, faces = octahedron()
    model = SpectralGeometry(embedding_dim=5).double()
    with torch.no_grad():
        model.conductivity_net[-1].weight.normal_(std=0.15)
    head = torch.nn.Linear(5, 3).double()
    normals, area, metric = normal_geometry(vertices, faces)
    assert not metric.requires_grad
    assert torch.linalg.eigvalsh(metric).min() > 0
    displacement = torch.zeros(len(vertices), dtype=vertices.dtype, requires_grad=True)
    moved = vertices + area.sqrt() * displacement[:, None] * normals
    embedding = model(moved, faces, *reference(vertices))
    survival = (1 - head(embedding).sigmoid()).cumprod(dim=0)
    penalty = 0.
    for value in survival:
        derivative = torch.autograd.grad(value, displacement, create_graph=True,
                                         retain_graph=True)[0]
        penalty = penalty + derivative @ torch.linalg.solve(metric, derivative) / len(survival)
    penalty.backward()
    assert torch.isfinite(penalty)
    assert penalty > 0
    assert model.conductivity_net[-1].weight.grad is not None
    assert torch.isfinite(model.conductivity_net[-1].weight.grad).all()
    assert model.conductivity_net[-1].weight.grad.abs().max() > 0
    assert model.scale_logits.grad is not None
    assert torch.isfinite(model.scale_logits.grad).all()
    assert head.weight.grad.abs().max() > 0


def test_invalid_mesh_and_sobolev_parameters_are_rejected():
    vertices, faces = octahedron()
    with pytest.raises(ValueError):
        sobolev_metric(vertices, faces, power=-1)
    with pytest.raises(ValueError):
        SpectralGeometry(scales=[[.1, .2], [.15, .3]])
    bad = vertices.clone()
    bad[2] = bad[0]
    with pytest.raises(ValueError):
        geometry_state(bad, faces)
