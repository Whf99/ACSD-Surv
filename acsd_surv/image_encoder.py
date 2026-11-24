"""Frozen MedicalNet encoder with deterministic content-addressed feature caching."""
import hashlib
import importlib.util
import os
from pathlib import Path
import torch
from torch import nn

ASSETS = Path(os.environ.get('MEDICALNET_ASSETS', 'assets/medicalnet')).resolve()

class MedicalEncoder(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=64):
        super().__init__()
        spec = importlib.util.spec_from_file_location('official_medicalnet', ASSETS/'resnet.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261111)
            self.backbone = module.resnet18(sample_input_D=48, sample_input_H=48,
                sample_input_W=48, num_seg_classes=2, shortcut_type='A', no_cuda=True)
        self.backbone.conv_seg = nn.Identity()
        if pretrained:
            checkpoint = torch.load(ASSETS/'resnet_18_23dataset.pth', map_location='cpu', weights_only=True)
            state = {k.removeprefix('module.'):v for k,v in checkpoint['state_dict'].items()
                     if not k.removeprefix('module.').startswith('conv_seg.')}
            self.backbone.load_state_dict(state, strict=True)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.project = nn.Linear(512, embedding_dim)
        identity=hashlib.sha256((ASSETS/'resnet.py').read_bytes())
        if pretrained:
            with (ASSETS/'resnet_18_23dataset.pth').open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''):identity.update(chunk)
        else:identity.update(b'random-seed-20261111')
        self.cache = Path(os.environ.get('ACSD_FEATURE_CACHE',str(ASSETS/'features'))) / identity.hexdigest()
        self.cache.mkdir(parents=True,exist_ok=True)
        self.memory = {}

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        features = []
        for image in images:
            key = hashlib.sha256(image.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            if key not in self.memory:
                path = self.cache/(key+'.pt')
                if path.exists():
                    value = torch.load(path, map_location='cpu', weights_only=True)
                else:
                    with torch.no_grad():
                        value = self.backbone(image[None]).mean(dim=(2,3,4))[0].detach().cpu()
                    assert value.shape == (512,) and torch.isfinite(value).all()
                    temporary = path.with_suffix('.'+str(os.getpid())+'.tmp')
                    torch.save(value, temporary)
                    temporary.replace(path)
                self.memory[key] = value
            features.append(self.memory[key])
        return self.project(torch.stack(features).to(images))
