"""Clinical anchor plus a zero-initialized multimodal log-risk residual."""
import torch
from torch import nn
from .geometry import SpectralGeometry


class ClinicalResidualPH(nn.Module):
    def __init__(self, clinical_dim, geometry_config, use_geometry=True,
                 use_image=True, geometry_embedding=16, image_embedding=64):
        super().__init__()
        self.use_geometry, self.use_image = use_geometry, use_image
        self.geo = (SpectralGeometry(embedding_dim=geometry_embedding,
                                    **geometry_config).double() if use_geometry else None)
        self.image = nn.Sequential(nn.Linear(512, image_embedding), nn.Tanh()) if use_image else None
        width = clinical_dim + geometry_embedding * use_geometry + image_embedding * use_image
        self.readout = nn.Sequential(nn.Linear(width, 32), nn.Softplus(), nn.Linear(32, 1, bias=False))
        nn.init.zeros_(self.readout[-1].weight)

    def scores(self, values, start, end, disable=False):
        q = values['q'][start:end]
        parts = [q]
        if self.use_geometry:
            parts.append(torch.stack([
                self.geo(m[0].double(), m[1], *(x.double() for x in m[2:])).to(q)
                for m in values['mesh'][start:end]
            ]))
        if self.use_image:
            parts.append(self.image(values['im'][start:end]))
        return values['base'][start:end] + self.readout(torch.cat(parts, 1)).ravel().double()
