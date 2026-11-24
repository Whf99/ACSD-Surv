"""Synthetic integration: real Cox fit, generated geometry, fixture image encoder."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import nibabel as nib

pytest.importorskip('sksurv')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from acsd_surv import training as run_fold


@pytest.mark.parametrize('arm',['H','G','E_S','E_eta'])
def test_real_runner_on_synthetic_images(tmp_path,arm,monkeypatch):
    torch.set_num_threads(1)
    data=tmp_path/'data';data.mkdir();splits=data/'splits/fold_1';splits.mkdir(parents=True)
    refs=[]
    for split_index,split in enumerate(['train','val','test']):
        rows=[]
        for i in range(12):
            pid='synthetic_%d_%d'%(split_index,i)
            image=np.random.default_rng(split_index*12+i).random((12,12,12)).astype(np.float32)
            grid=np.indices(image.shape);mask=(((grid-5.5)**2).sum(0)<(2+i%3*.2)**2).astype(np.uint8)
            nib.save(nib.Nifti1Image(image,np.eye(4)),str(data/(pid+'_image.nii.gz')))
            nib.save(nib.Nifti1Image(mask,np.eye(4)),str(data/(pid+'_mask.nii.gz')))
            rows.append(dict(patient_id=pid,stage=1+i%4,time=i+1,recur_label=int(i%3!=1),
                             image_path=pid+'_image.nii.gz',label_path=pid+'_mask.nii.gz'))
            refs.append(dict(patient_id=pid,origin_x=0,origin_y=0,origin_z=0,length_mm=100))
        with (splits/(split+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    reference=data/'references.csv'
    with reference.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=refs[0]);writer.writeheader();writer.writerows(refs)
    cfg=json.loads((ROOT/'configs/NPC.json').read_text())
    cfg.update(data_root=str(data),split_root=str(data/'splits'),cache_dir=str(tmp_path/'cache'),
               expected_cohort_size=36,evaluation_grid=[2,3,4])
    cfg['data_provenance']={
        'time_origin_verified':True,'time_origin_evidence':'synthetic fixture, not clinical evidence',
        'anatomical_reference_verified':True,'anatomical_reference_evidence':'synthetic fixture, not clinical evidence'}
    cfg['preprocess'].update(reference_csv=str(reference),shape=[8,8,8],crop_mm=[12,12,12],mesh_max_vertices=64)
    cfg['training'].update(epochs=2,patience=2,batch_size=4,replay_block_size=3,
                           geometry_rho=.001 if arm in ('E_S','E_eta') else 0)
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg),encoding='utf-8')
    # Exercise the external-asset loading interface without downloading or
    # claiming validation of real MedicalNet source/weights.
    from acsd_surv import image_encoder as medical_encoder
    assets=tmp_path/'synthetic_encoder';assets.mkdir()
    (assets/'resnet.py').write_text('''from torch import nn
class SyntheticBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv3d(1,512,1);self.conv_seg=nn.Identity()
    def forward(self,x):return self.conv_seg(self.conv(x))
def resnet18(**kwargs):return SyntheticBackbone()
''',encoding='utf-8')
    torch.save({'state_dict':{'conv.weight':torch.full((512,1,1,1,1),.01),
                              'conv.bias':torch.zeros(512)}},assets/'resnet_18_23dataset.pth')
    monkeypatch.setattr(medical_encoder,'ASSETS',assets)
    run_fold.CONFIG_FILE=config;run_fold.HOME_STUDY=tmp_path/'outputs'
    run_fold.worker('NPC',1,arm)
    dest=tmp_path/'outputs/NPC/fold_1'/arm
    result=json.loads((dest/'result.json').read_text())
    assert all(np.isfinite(x) for x in result['metrics'].values())
    assert json.loads((dest/'effective_config.json').read_text())['training']['lr']==cfg['training']['lr']
    inference=np.load(dest/'inference_state.npz')
    assert inference['baseline_hazard'].shape==(3,)
    if arm in ('E_S','E_eta'):
        diagnostics=json.loads((dest/'regularization.json').read_text())
        assert all(row['rho']==.001 for row in diagnostics)


def test_feature_cache_distinguishes_pretrained_weights(tmp_path,monkeypatch):
    from acsd_surv import image_encoder
    source='''from torch import nn
class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv3d(1,512,1);self.conv_seg=nn.Identity()
    def forward(self,x):return self.conv_seg(self.conv(x))
def resnet18(**kw):return Backbone()
'''
    (tmp_path/'resnet.py').write_text(source)
    monkeypatch.setattr(image_encoder,'ASSETS',tmp_path)
    def save(value):
        torch.save({'state_dict':{'conv.weight':torch.full((512,1,1,1,1),value),
                                  'conv.bias':torch.zeros(512)}},tmp_path/'resnet_18_23dataset.pth')
    save(.01);first=image_encoder.MedicalEncoder()
    save(.02);second=image_encoder.MedicalEncoder()
    assert first.cache!=second.cache
    first.project=second.project=torch.nn.Identity()
    image=torch.ones(1,1,2,2,2)
    torch.testing.assert_close(second(image),2*first(image))
