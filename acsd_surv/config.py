"""Effective configuration and clinical provenance checks."""
import math
import torch

def quadrature_weights(grid, supplied=None):
    t=torch.as_tensor(grid,dtype=torch.float64)
    if t.ndim!=1 or len(t)<2 or not torch.isfinite(t).all() or t[0]<=0 or not (t[1:]>t[:-1]).all():
        raise ValueError('evaluation_grid must be finite, positive and strictly increasing')
    if supplied is None:
        dt=t[1:]-t[:-1]
        w=torch.cat([dt[:1]/2,(dt[:-1]+dt[1:])/2,dt[-1:]/2])/(t[-1]-t[0])
    else:
        w=torch.as_tensor(supplied,dtype=torch.float64)
    if w.shape!=t.shape or not torch.isfinite(w).all() or not (w>0).all() or abs(float(w.sum())-1)>1e-10:
        raise ValueError('horizon_weights must be positive and sum to one')
    return w

def validate_config(cfg,arm):
    t=cfg['training']; geo=cfg['model']['geometry']
    quadrature_weights(cfg['evaluation_grid'],cfg.get('horizon_weights'))
    if not 0<cfg['ipcw_clip']<=1 or not math.isfinite(cfg['clinical_anchor_alpha']) or cfg['clinical_anchor_alpha']<0:
        raise ValueError('Invalid IPCW clip or clinical anchor ridge coefficient')
    if t['optimizer']!='Adam' or t['loss']!='cox_plus_ipcw_brier' or t['regularization_schedule']!='fixed':
        raise ValueError('Training uses Adam with explicit L2, Cox+Brier and fixed rho')
    for key in ('lr','l2_lambda','geometry_rho','grad_clip'):
        if not math.isfinite(t[key]) or t[key]<0:raise ValueError('Invalid '+key)
    if t['lr']==0:raise ValueError('lr must be positive')
    for key in ('epochs','patience','replay_block_size','batch_size','cpu_threads'):
        if not isinstance(t[key],int) or t[key]<1:raise ValueError('Invalid '+key)
    if arm in ('E_S','E_eta') and t['geometry_rho']<=0:
        raise ValueError('Set an explicit positive training.geometry_rho for E_S/E_eta')
    if arm not in ('E_S','E_eta') and t['geometry_rho']!=0:
        raise ValueError('Unregularized arms require geometry_rho=0')
    if not geo.get('include_position') or not geo.get('include_size'):
        raise ValueError('The representation requires position and size: 11*m+10')
    provenance=cfg.get('data_provenance',{})
    if cfg.get('time_origin')!='treatment_initiation' or not provenance.get('time_origin_verified') or not provenance.get('time_origin_evidence'):
        raise ValueError('Verify treatment-initiation time origin and record its evidence before real-data training')
    if arm not in ('H','A') and (not cfg['preprocess'].get('reference_csv') or not provenance.get('anatomical_reference_verified') or not provenance.get('anatomical_reference_evidence')):
        raise ValueError('Provide verified cranio-cervical reference_csv and extraction evidence; FOV center is not a verified anatomical origin')
    power=cfg['model']['sobolev_power'];ell=cfg['model']['sobolev_ell']
    if not math.isfinite(power) or power<0 or not math.isfinite(ell) or ell<=0:
        raise ValueError('Invalid Sobolev metric parameters')
    if t['device']!='cpu':raise ValueError('This dense reference runner currently supports device=cpu')


def load_config(path):
    """Resolve data/asset paths relative to the configuration file."""
    import json
    from pathlib import Path
    path=Path(path).resolve()
    cfg=json.loads(path.read_text(encoding='utf-8'))
    for key in ('data_root','split_root','cache_dir','output_root'):
        if cfg.get(key):
            p=Path(cfg[key]);cfg[key]=str(p if p.is_absolute() else (path.parent/p).resolve())
    for container,key in ((cfg['preprocess'],'reference_csv'),(cfg['model']['geometry'],'isotropic_cache_dir')):
        if container.get(key):
            p=Path(container[key]);container[key]=str(p if p.is_absolute() else (path.parent/p).resolve())
    return cfg
