"""Dense differentiable FEM reference for the anatomical spectral branch.

Uses area-normalized finite-element surface operators. Topology and the
external anatomical reference are fixed inputs, while every geometric quantity
is recomputed from the current vertices. Full matrix exponentials deliberately
avoid derivatives of individual eigenvectors at repeated eigenvalues. This is
a reference implementation for small meshes, not a mesh-convergence claim.

Use consistently oriented, closed, connected, nondegenerate triangle meshes.
Double precision is recommended for derivative checks and Sobolev solves.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import hashlib
from pathlib import Path
import torch
from torch import Tensor, nn


def _validate_mesh(vertices: Tensor, faces: Tensor) -> Tensor:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError("vertices must have shape [V, 3] with V >= 4")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
        raise ValueError("faces must have shape [F, 3] with F >= 4")
    if not vertices.is_floating_point():
        raise TypeError("vertices must be floating point")
    faces = faces.to(device=vertices.device, dtype=torch.long)
    if bool((faces < 0).any()) or bool((faces >= len(vertices)).any()):
        raise ValueError("face index outside vertices")
    return faces


def _assemble_stiffness(
    faces: Tensor, face_areas: Tensor, gradients: Tensor,
    n_vertices: int, conductivity: Optional[Tensor] = None,
) -> Tensor:
    if conductivity is None:
        local = torch.einsum("fia,fja->fij", gradients, gradients)
    else:
        local = torch.einsum("fia,fab,fjb->fij", gradients, conductivity, gradients)
    local = face_areas[:, None, None] * local
    rows = faces[:, :, None].expand(-1, 3, 3)
    cols = faces[:, None, :].expand(-1, 3, 3)
    indices = (rows * n_vertices + cols).reshape(-1)
    # Out-of-place index_add preserves vertex and mixed parameter derivatives.
    flat = local.new_zeros(n_vertices * n_vertices).index_add(
        0, indices, local.reshape(-1)
    )
    matrix = flat.reshape(n_vertices, n_vertices)
    return 0.5 * (matrix + matrix.transpose(0, 1))


def _normalized_operator(stiffness: Tensor, mass: Tensor, area: Tensor) -> Tensor:
    inv_sqrt_mass = mass.rsqrt()
    operator = area * inv_sqrt_mass[:, None] * stiffness * inv_sqrt_mass[None, :]
    return 0.5 * (operator + operator.transpose(0, 1))


def geometry_state(vertices: Tensor, faces: Tensor) -> Dict[str, Tensor]:
    """Recompute mass, curvature, normals and isotropic FEM operator.

    The area is in squared physical-coordinate units. The operator is area
    normalized. ``normals`` are area-weighted vertex normals. All floating
    outputs retain the vertices' device, dtype and autograd graph.
    """
    faces = _validate_mesh(vertices, faces)
    triangle = vertices[faces]
    cross = torch.cross(triangle[:, 1] - triangle[:, 0],
                        triangle[:, 2] - triangle[:, 0], dim=-1)
    double_areas = torch.linalg.vector_norm(cross, dim=-1)
    if not bool(torch.isfinite(double_areas).all()) or bool((double_areas <= 0).any()):
        raise ValueError("mesh contains degenerate or non-finite triangles")
    face_areas = double_areas * 0.5
    face_normals = cross / double_areas[:, None]
    area = face_areas.sum()
    indices = faces.reshape(-1)
    mass = vertices.new_zeros(len(vertices)).index_add(
        0, indices, (face_areas[:, None].expand(-1, 3) / 3).reshape(-1)
    )
    if bool((mass <= 0).any()):
        raise ValueError("mesh contains unused vertices or zero vertex mass")
    weighted_normals = vertices.new_zeros(vertices.shape).index_add(
        0, indices, cross[:, None, :].expand(-1, 3, -1).reshape(-1, 3)
    )
    normal_lengths = torch.linalg.vector_norm(weighted_normals, dim=-1)
    if bool((normal_lengths <= 0).any()):
        raise ValueError("mesh contains undefined vertex normals; check face orientation")
    normals = weighted_normals / normal_lengths[:, None]
    gradients = torch.stack([
        torch.cross(face_normals, triangle[:, 2] - triangle[:, 1], dim=-1),
        torch.cross(face_normals, triangle[:, 0] - triangle[:, 2], dim=-1),
        torch.cross(face_normals, triangle[:, 1] - triangle[:, 0], dim=-1),
    ], dim=1) / double_areas[:, None, None]
    stiffness = _assemble_stiffness(faces, face_areas, gradients, len(vertices))
    operator = _normalized_operator(stiffness, mass, area)
    centroid = (mass[:, None] * vertices).sum(dim=0) / area
    curvature_vector = (stiffness @ vertices) / mass[:, None]
    curvature = 0.5 * (normals * curvature_vector).sum(dim=-1)
    return {
        "faces": faces, "area": area, "mass": mass, "normals": normals,
        "face_areas": face_areas, "face_normals": face_normals,
        "face_gradients": gradients, "face_centroids": triangle.mean(dim=1),
        "stiffness": stiffness, "isotropic_operator": operator,
        "centroid": centroid, "curvature": curvature,
    }


def sobolev_metric(vertices: Tensor, faces: Tensor, ell: float = 0.1,
                   power: float = 4) -> Tensor:
    """Return R = D (I + ell**2 S0)**power D for a baseline mesh.

    ``power`` may be any finite nonnegative exponent; power > 3 is needed
    for the C2 regularity bound, not for defining the metric. Use a solve, not an explicit matrix inverse.
    Detach this metric when applying a prediction-sensitivity penalty: its value
    must not be learned or recomputed on the perturbed mesh.
    """
    if ell <= 0 or isinstance(power, bool) or not torch.isfinite(torch.tensor(float(power))) or power < 0:
        raise ValueError("ell must be positive and power must be finite and nonnegative")
    state = geometry_state(vertices, faces)
    identity = torch.eye(len(vertices), dtype=vertices.dtype, device=vertices.device)
    base = identity + ell * ell * state["isotropic_operator"]
    if int(power) == power:
        powered = torch.linalg.matrix_power(base, int(power))
    else:
        values, vectors = torch.linalg.eigh(base)
        powered = (vectors * values.pow(power)[None, :]) @ vectors.T
    d = torch.sqrt(state["mass"] / state["area"])
    metric = d[:, None] * powered * d[None, :]
    return 0.5 * (metric + metric.transpose(0, 1))


def normal_geometry(vertices: Tensor, faces: Tensor, ell: float = 0.1,
                    power: int = 4) -> Tuple[Tensor, Tensor, Tensor]:
    """Frozen baseline ``(normals, area, R)`` for normal displacements.

    Differentiate predictions at ``v + sqrt(area) * u[:,None] * normals`` with
    respect to dimensionless ``u`` at zero. With ``gamma = dS/du``, the penalty
    is ``gamma @ torch.linalg.solve(R, gamma)``. Training this penalty requires
    ``create_graph=True`` in the first autograd call.
    """
    with torch.no_grad():
        state = geometry_state(vertices.detach(), faces)
        metric = (state["mass"] / state["area"] if power == 0 else
                  sobolev_metric(vertices.detach(), faces, ell, power))
    return state["normals"], state["area"], metric


class SpectralGeometry(nn.Module):
    """Learn anatomy-conditioned diffusion and compact spectral readout.

    Anatomical axes are columns ordered right, anterior, superior in physical
    coordinates. ``reference_origin`` must come from an explicitly supplied
    anatomical landmark/atlas reference; it is not inferred from the tumor.
    Reference extraction and topology generation remain preprocessing steps.
    """

    def __init__(
        self, scales: Optional[Sequence[Sequence[float]]] = None,
        embedding_dim: int = 16, conductivity_min: float = 0.5,
        conductivity_max: float = 2.0, sigma: float = 0.03,
        kappa: float = 8.0, learn_operator: bool = True,
        learn_scales: bool = True, include_position: bool = True, include_size: bool = True,
        isotropic_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        if scales is None:
            scales = ((0.01, 0.03), (0.04, 0.1), (0.15, 0.4))
        bounds = torch.as_tensor(scales, dtype=torch.get_default_dtype())
        if (bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) == 0
                or bool((bounds <= 0).any())
                or bool((bounds[:, 1] <= bounds[:, 0]).any())
                or bool((bounds[1:, 0] <= bounds[:-1, 1]).any())):
            raise ValueError("scales require strictly ordered disjoint positive intervals")
        if not 0 < conductivity_min <= conductivity_max or sigma <= 0 or kappa <= 0:
            raise ValueError("invalid conductivity bounds, smoothing time or window width")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.register_buffer("scale_bounds", bounds)
        self.scale_logits = nn.Parameter(torch.zeros(len(bounds)), requires_grad=learn_scales)
        self.conductivity_min = float(conductivity_min)
        self.conductivity_max = float(conductivity_max)
        self.sigma = float(sigma)
        self.kappa = float(kappa)
        self.learn_operator = bool(learn_operator)
        self.include_position = bool(include_position)
        self.include_size = bool(include_size)
        self.isotropic_cache_dir = Path(isotropic_cache_dir) if isotropic_cache_dir else None
        self._isotropic_eigen_cache = {}
        self.raw_dim = 11 * len(bounds) + (9 if self.include_position else 6) + int(self.include_size)
        self.embedding_dim = int(embedding_dim)
        self.conductivity_net = nn.Sequential(nn.Linear(6, 24), nn.Tanh(), nn.Linear(24, 3))
        # Constant equal initial conductivities recover exactly the isotropic L0.
        nn.init.zeros_(self.conductivity_net[-1].weight)
        nn.init.zeros_(self.conductivity_net[-1].bias)
        for parameter in self.conductivity_net.parameters():
            parameter.requires_grad_(self.learn_operator)
        self.readout = nn.Sequential(
            nn.Linear(self.raw_dim, 32), nn.Tanh(), nn.Linear(32, embedding_dim), nn.Tanh()
        )
        upper = torch.triu_indices(4, 4)
        self.register_buffer("upper_rows", upper[0])
        self.register_buffer("upper_cols", upper[1])

    def diffusion_scales(self) -> Tensor:
        log_bounds = self.scale_bounds.log()
        return torch.exp(log_bounds[:, 0] + self.scale_logits.sigmoid()
                         * (log_bounds[:, 1] - log_bounds[:, 0]))

    def _operator(self, vertices: Tensor, state: Dict[str, Tensor], axes: Tensor) -> Tensor:
        if not self.learn_operator:
            return state["isotropic_operator"]
        xi = ((state["face_centroids"] - state["centroid"]) @ axes
              / state["area"].sqrt())
        anatomical_normals = state["face_normals"] @ axes
        inputs = torch.cat((xi, anatomical_normals), dim=-1)
        network_dtype = self.conductivity_net[0].weight.dtype
        logits = self.conductivity_net(inputs.to(dtype=network_dtype))
        coefficients = (self.conductivity_min
                        + (self.conductivity_max - self.conductivity_min) * logits.sigmoid())
        coefficients = coefficients.to(dtype=vertices.dtype)
        normals = state["face_normals"]
        identity = torch.eye(3, device=vertices.device, dtype=vertices.dtype)
        projection = identity[None] - normals[:, :, None] * normals[:, None, :]
        anatomical_tensor = torch.einsum("ia,fa,ja->fij", axes, coefficients, axes)
        unnormalized = projection @ anatomical_tensor @ projection
        trace = torch.diagonal(unnormalized, dim1=-2, dim2=-1).sum(dim=-1)
        mean_trace = (state["face_areas"] * trace).sum() / (2 * state["area"])
        conductivity = unnormalized / mean_trace
        stiffness = _assemble_stiffness(
            state["faces"], state["face_areas"], state["face_gradients"],
            len(vertices), conductivity,
        )
        return _normalized_operator(stiffness, state["mass"], state["area"])

    def operator(self, vertices: Tensor, faces: Tensor, reference_axes: Tensor) -> Tensor:
        """Return the learned area-normalized symmetric FEM operator."""
        axes = torch.as_tensor(reference_axes, device=vertices.device, dtype=vertices.dtype)
        if axes.shape != (3, 3):
            raise ValueError("reference_axes must have shape [3,3]")
        return self._operator(vertices, geometry_state(vertices, faces), axes)

    def _responses(
        self, vertices: Tensor, faces: Tensor, reference_origin: Tensor,
        reference_axes: Tensor, reference_length: Union[float, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        origin = torch.as_tensor(reference_origin, device=vertices.device, dtype=vertices.dtype)
        axes = torch.as_tensor(reference_axes, device=vertices.device, dtype=vertices.dtype)
        length = torch.as_tensor(reference_length, device=vertices.device, dtype=vertices.dtype)
        if origin.shape != (3,) or axes.shape != (3, 3) or length.numel() != 1:
            raise ValueError("reference shapes must be [3], [3,3] and scalar")
        if not bool(torch.isfinite(length)) or bool(length <= 0):
            raise ValueError("reference_length must be finite and positive")
        state = geometry_state(vertices, faces)
        operator = self._operator(vertices, state, axes)
        sqrt_area = state["area"].sqrt()
        sqrt_measure = (state["mass"] / state["area"]).sqrt()
        xi = (vertices - state["centroid"]) @ axes / sqrt_area
        # Fixed geometry is diagonalized offline/in memory only when no shape
        # derivative is requested. Perturbed meshes retain matrix-exp autograd.
        eigen = None
        if not self.learn_operator and not vertices.requires_grad:
            digest = hashlib.sha256(b'acsd-isotropic-eigen-v1')
            for tensor in (vertices, faces):
                digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            key = digest.hexdigest()
            if key not in self._isotropic_eigen_cache:
                path = self.isotropic_cache_dir / (key+'.pt') if self.isotropic_cache_dir else None
                if path is not None and path.exists():
                    eigen = torch.load(path, map_location=vertices.device, weights_only=True)
                else:
                    eigen = tuple(x.detach() for x in torch.linalg.eigh(operator))
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(tuple(x.cpu() for x in eigen), path)
                self._isotropic_eigen_cache[key] = eigen
            eigen = tuple(x.to(vertices) for x in self._isotropic_eigen_cache[key])
        def heat_at(scale):
            if eigen is None:
                return torch.matrix_exp(-scale * operator)
            values, vectors = eigen
            return (vectors * torch.exp(-scale * values)[None, :]) @ vectors.T
        field = heat_at(self.sigma) @ (
            sqrt_measure * sqrt_area * state["curvature"]
        )
        windows = torch.sigmoid(self.kappa * xi)
        fields = torch.cat((field[:, None], windows * field[:, None]), dim=-1)
        responses = []
        grams = []
        for scale in self.diffusion_scales().to(dtype=vertices.dtype):
            heat = heat_at(scale)
            gram = fields.transpose(0, 1) @ heat @ fields
            gram = 0.5 * (gram + gram.transpose(0, 1))
            heat_trace = heat.diagonal().sum() - 1
            responses.append(torch.cat((heat_trace.reshape(1), gram[self.upper_rows, self.upper_cols])))
            grams.append(gram)
        # Six labeled extents ordered (+R,-R,+A,-A,+S,-S).
        directions = torch.stack((xi, -xi), dim=-1).reshape(len(vertices), 6)
        log_measure = (state["mass"] / state["area"]).log()
        extents = torch.logsumexp(
            self.kappa * directions + log_measure[:, None], dim=0
        ) / self.kappa
        position = (state["centroid"] - origin) @ axes / length
        raw = torch.cat((*responses, extents, position)) if self.include_position else torch.cat((*responses, extents))
        if self.include_size:
            raw = torch.cat((raw, (sqrt_area / length).log().reshape(1)))
        return raw, tuple(grams)

    def features(
        self, vertices: Tensor, faces: Tensor, reference_origin: Tensor,
        reference_axes: Tensor, reference_length: Union[float, Tensor],
    ) -> Tensor:
        """Raw responses, 11*m+10 with position and size enabled, before neural readout."""
        return self._responses(vertices, faces, reference_origin, reference_axes, reference_length)[0]

    def response_matrices(
        self, vertices: Tensor, faces: Tensor, reference_origin: Tensor,
        reference_axes: Tensor, reference_length: Union[float, Tensor],
    ) -> Tuple[Tensor, ...]:
        """Four-field heat Gram matrices, exposed for numerical verification."""
        return self._responses(vertices, faces, reference_origin, reference_axes, reference_length)[1]

    def forward(
        self, vertices: Tensor, faces: Tensor, reference_origin: Tensor,
        reference_axes: Tensor, reference_length: Union[float, Tensor],
    ) -> Tensor:
        raw = self.features(vertices, faces, reference_origin, reference_axes, reference_length)
        # Smooth compression controls raw trace/energy scales without estimating
        # any population statistics. asinh, unlike sign*log1p, is smooth at zero.
        compressed = torch.asinh(raw).to(dtype=self.readout[0].weight.dtype)
        return self.readout(compressed)


def metric_quadratic(gradient, metric):
    """Diagonal mass scaling at r_s=0; one dense solve otherwise."""
    if metric.ndim == 1:
        return (gradient.square() / metric).sum()
    return gradient @ torch.linalg.solve(metric, gradient)

def six_direction_responses(gram):
    transform = gram.new_tensor([[0,1,0,0],[1,-1,0,0],[0,0,1,0],
                                 [1,0,-1,0],[0,0,0,1],[1,0,0,-1]])
    return transform @ gram @ transform.T
