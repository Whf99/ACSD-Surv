"""Full-risk-set clinical-residual proportional-hazards training."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HOME_STUDY=ROOT/'outputs'
CONFIG_FILE=None
def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(path)

def worker(ds,fold,arm,pilot=False):
    import torch
    from torch import nn
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    from acsd_surv.data import adapter,make_loader as data_loader
    from acsd_surv.utils import seed_all
    load_fold=adapter(ds).load_fold
    def make_loader(rows,config):return data_loader(rows,config,ds)
    from acsd_surv.config import validate_config,quadrature_weights
    from acsd_surv.censoring import CensorKM
    from acsd_surv.regularization import reg_step, perturbation
    from acsd_surv.image_encoder import MedicalEncoder
    from acsd_surv.metrics import evaluate_fold
    from acsd_surv.config import load_config
    cfg=load_config(CONFIG_FILE)
    validate_config(cfg,arm)
    torch.set_num_threads(cfg['training']['cpu_threads'])
    folder=HOME_STUDY/ds/f'fold_{fold}'
    folder.mkdir(parents=True,exist_ok=True)
    rows,audit=load_fold(Path(cfg['data_root']),Path(cfg['split_root']),fold,cfg['zero_time_policy'])
    months=np.array(cfg['evaluation_grid'],dtype=float)
    weights=quadrature_weights(months,cfg.get('horizon_weights'))
    if ds=='HECKTOR' and cfg.get('mask_definition')!='primary_tumor':
        raise ValueError('HECKTOR requires primary tumor masks')
    use_g=arm not in ('H','A')
    cfg['preprocess']['need_geometry']=use_g
    use_im=arm!='H' and arm!='G'
    encoder=MedicalEncoder(pretrained=True).eval() if use_im else None
    if encoder is not None:encoder.project=nn.Identity()
    def get(split):
        images=[];clinical=[];meshes=[]; ids=[]
        with torch.no_grad():
            for b in make_loader(rows[split],cfg):
                images.append(encoder(b['image']).detach() if use_im else torch.empty(len(b['clinical']),0))
                clinical.append(b['clinical'])
                ids.extend(b['patient_id'])
                if use_g:meshes.extend(zip(*(b[k] for k in ('vertices','faces','reference_origin','reference_axes','reference_length'))))
        assert ids==[r['patient_id'] for r in rows[split]]
        rr=rows[split]
        return dict(im=torch.cat(images),q=torch.cat(clinical),mesh=meshes,ids=ids,
                    t=np.array([float(r['time']) for r in rr]),e=np.array([int(r['recur_label']) for r in rr]))
    data={s:get(s) for s in ('train','val')}
    tr,va=data.values()
    if cfg.get('expected_cohort_size') is not None:
        assert sum(len(rr) for rr in rows.values())==cfg['expected_cohort_size']
    def tensor(x):return torch.as_tensor(x,dtype=torch.float64)
    scale=StandardScaler().fit(tr['q'].numpy())
    x=scale.transform(tr['q'].numpy())
    cox=CoxPHSurvivalAnalysis(alpha=cfg['clinical_anchor_alpha'],ties='breslow').fit(x,Surv.from_arrays(tr['e'].astype(bool),tr['t']))
    center=cox.predict(x).mean()
    censor=CensorKM().fit(tr['t'],tr['e'])
    for v in data.values():
        v['base']=tensor(cox.predict(scale.transform(v['q'].numpy()))-center)
        t=v['t'][:,None];e=v['e'][:,None]
        v['alive']=tensor((t>months)/np.maximum(cfg['ipcw_clip'],censor.predict(months)))
        v['dead']=tensor((t<=months)*e/np.maximum(cfg['ipcw_clip'],censor.predict(t,left=True)))
    from acsd_surv.survival import breslow_hazard,cox_loss,ph_survival,weighted_brier,horizon_support
    train_t=tensor(tr['t']);train_e=tensor(tr['e']);eval_grid=tensor(months)
    support={name:horizon_support(tensor(v['t']),tensor(v['e']),eval_grid) for name,v in data.items()}
    def hazard(z):return breslow_hazard(z,train_t,train_e,eval_grid)
    def curve(mode,z,ztr,bias):return ph_survival(z,hazard(ztr))
    def brier(s,v):return weighted_brier(s,v['alive'],v['dead'],weights)
    def objective(z,bias,mode):return cox_loss(z,train_t,train_e)+brier(curve(mode,z,z,bias),tr)
    # Baseline parity on held-out clinical features uses no held-out outcomes.
    expected=np.array([f(months) for f in cox.predict_survival_function(scale.transform(va['q'].numpy()))])
    err=float(abs(expected-curve('ph',va['base'],tr['base'],None).numpy()).max())
    assert err<1e-8,err
    from acsd_surv.model import ClinicalResidualPH
    def Model(design):
        return ClinicalResidualPH(tr['q'].shape[1],cfg['model']['geometry'],use_g,use_im,
                                  cfg['model']['geometry_embedding'],cfg['model']['image_embedding'])
    if True:
        dest=folder/(('pilot_' if pilot else '')+arm);dest.mkdir(exist_ok=True)
        if any(dest.iterdir()):raise FileExistsError('Choose a new empty run directory: '+str(dest))
        mode='ph'
        seed_all(cfg['seed']+fold)
        model=Model('concat')
        bias=nn.Parameter(torch.zeros(1,dtype=torch.float64),requires_grad=False)
        opt=torch.optim.Adam(model.parameters(),lr=cfg['training']['lr'],weight_decay=0.)
        block=cfg['training']['replay_block_size']
        def blocks(v):return [(i,min(i+block,len(v['ids']))) for i in range(0,len(v['ids']),block)]
        @torch.no_grad()
        def scores(v,disable=False):return torch.cat([model.scores(v,a,b,disable) for a,b in blocks(v)])
        # Numerical parity on the actual risk-set objective and replayed parameter gradients.
        a,b=0,min(block,len(tr['ids']))
        probe=scores(tr).detach()
        direct=model.scores(tr,a,b)
        obj=objective(torch.cat([direct,probe[b:]]),bias,mode)
        pars=[p for p in model.parameters() if p.requires_grad]+([bias] if bias.requires_grad else [])
        direct_grad=torch.autograd.grad(obj,pars,allow_unused=True)
        trial=probe.clone().requires_grad_()
        obj=objective(trial,bias,mode)
        g=torch.autograd.grad(obj,[trial]+([bias] if bias.requires_grad else []))
        local=model.scores(tr,a,b)@g[0][a:b].detach()
        if bias.requires_grad:local=local+bias@g[1].detach()
        replay=torch.autograd.grad(local,pars,allow_unused=True)
        assert all(torch.allclose(v,w,rtol=1e-5,atol=1e-8) for v,w in zip(direct_grad,replay) if v is not None)
        best=float('inf');history=[]
        z=scores(tr)
        diag=[];cache={}
        regular=arm in ('E_S','E_eta')
        write(dest/'effective_config.json',cfg)
        write(dest/'training_validation_horizon_support.json',support)
        for epoch in range(cfg['training']['epochs']+1):
            if epoch:
                z=z.requires_grad_()
                loss=objective(z,bias,mode)
                opt.zero_grad()
                loss.backward()
                for a,b in blocks(tr):
                    (model.scores(tr,a,b)@z.grad[a:b]).backward()
                if regular:
                    stats=reg_step(model,tr,z.detach(),hazard(z.detach()).detach(),arm,cfg['training']['geometry_rho'],cache,weights,cfg['model']['sobolev_ell'],cfg['model']['sobolev_power'])
                    stats['epoch']=epoch
                    diag.append(stats);write(dest/'regularization.json',diag)
                # Explicit lambda*||Theta||^2 gradient; no decoupled weight decay.
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
                        parameter.grad.add_(parameter.detach(),alpha=2*cfg['training']['l2_lambda'])
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['grad_clip'])
                opt.step()
                z=scores(tr)
            with torch.no_grad():val=float(brier(curve(mode,scores(va),z,bias),va))
            history.append(val)
            if val<best:
                best=val;selected=epoch
                torch.save((model.state_dict(),bias.detach().clone()),dest/'selected.pt')
            write(dest/'history.json',history)
            if epoch-selected>=cfg['training']['patience']:break
        state,saved_bias=torch.load(dest/'selected.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state)
        # Save the clinical transform and the selected training baseline for inference.
        with torch.no_grad():
            np.savez_compressed(dest/'inference_state.npz',clinical_mean=scale.mean_,
                clinical_scale=scale.scale_,clinical_coef=cox.coef_,clinical_center=center,
                evaluation_grid=months,baseline_hazard=hazard(scores(tr)).numpy())
        te=get('test')
        te['base']=tensor(cox.predict(scale.transform(te['q'].numpy()))-center)
        def evaluate(disable):
            with torch.no_grad():s=curve(mode,scores(te,disable),scores(tr),saved_bias).numpy()
            m=evaluate_fold(tr['t'],tr['e'],te['t'],te['e'],None,s,months,common_grid=months)
            write(dest/'evaluation_details.json',m)
            np.savez_compressed(dest/('image_off.npz' if disable else 'test.npz'),patient_id=te['ids'],time=te['t'],event=te['e'],survival=s,grid=months)
            return {k:m[k] for k in ('Cindex','IAUC','IBS')}
        metrics=evaluate(False)
        off=None
        sd=float((scores(tr)-tr['base']).std())
        if regular:
            with torch.no_grad():
                perturbation(model,te,months,hazard(scores(tr)),dest,weights=weights)
        write(dest/'result.json',dict(dataset=ds,fold=fold,arm=arm,selected=selected,metrics=metrics,train_sd_correction=sd,near_zero=sd<.01,anchor_error=err))
        print(ds,fold,arm,metrics,flush=True)

if __name__=='__main__':
    raise SystemExit('Use scripts/run_study.py')
