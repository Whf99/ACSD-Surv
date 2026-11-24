import copy
import json
import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from acsd_surv.geometry import SpectralGeometry, normal_geometry, metric_quadratic, geometry_state
from acsd_surv.config import quadrature_weights,validate_config
from acsd_surv.model import ClinicalResidualPH
from acsd_surv.survival import breslow_hazard,cox_loss,ph_survival,weighted_brier
from acsd_surv.regularization import reg_step


def mesh():
    v=torch.tensor([[1.4,0,0],[-1.1,0,0],[0,1.2,0],[0,-.9,0],[0,0,1.6],[0,0,-.8]],dtype=torch.float64)
    f=torch.tensor([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
    return v,f,torch.zeros(3,dtype=v.dtype),torch.eye(3,dtype=v.dtype),v.new_tensor(100.)


def test_diagonal_metric_avoids_linear_solve(monkeypatch):
    m=mesh();normal,area,metric=normal_geometry(m[0],m[1],power=0)
    state=geometry_state(m[0],m[1]);g=torch.arange(1,7,dtype=m[0].dtype)
    assert metric.ndim==1
    expected=(g.square()/(state['mass']/area)).sum()
    def forbidden(*a,**kw):raise AssertionError('r_s=0 must not solve a linear system')
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    torch.testing.assert_close(metric_quadratic(g,metric),expected)


def test_offline_isotropic_cache_preserves_values_and_scale_gradients(tmp_path):
    m=mesh();net=SpectralGeometry(learn_operator=False,isotropic_cache_dir=str(tmp_path)).double()
    raw=net.features(*m);grad=torch.autograd.grad(raw.square().sum(),net.scale_logits)[0]
    moved=(m[0].clone().requires_grad_(),*m[1:])
    direct=net.features(*moved)
    direct_grad=torch.autograd.grad(direct.square().sum(),net.scale_logits,retain_graph=True)[0]
    torch.testing.assert_close(raw,direct,rtol=1e-9,atol=1e-9)
    torch.testing.assert_close(grad,direct_grad,rtol=1e-8,atol=1e-8)
    assert torch.isfinite(torch.autograd.grad(direct.sum(),moved[0])[0]).all()
    assert len(list(tmp_path.glob('*.pt')))==1


def test_quadrature_and_clinical_provenance_are_explicit():
    torch.testing.assert_close(quadrature_weights([1,2,4]),torch.tensor([1/6,1/2,1/3],dtype=torch.float64))
    cfg=json.loads((ROOT/'configs/NPC.json').read_text())
    with pytest.raises(ValueError,match='treatment-initiation'):validate_config(cfg,'D')
    cfg['data_provenance'].update(time_origin_verified=True,time_origin_evidence='synthetic test only')
    with pytest.raises(ValueError,match='cranio-cervical'):validate_config(cfg,'D')
    cfg['preprocess']['reference_csv']='synthetic_reference.csv'
    cfg['data_provenance'].update(anatomical_reference_verified=True,anatomical_reference_evidence='synthetic test only')
    validate_config(cfg,'D')
    with pytest.raises(ValueError,match='positive'):validate_config(cfg,'E_eta')


def test_breslow_ties_and_training_baseline_derivative():
    z=torch.tensor([.1,-.2,.3,.4],dtype=torch.float64,requires_grad=True)
    t=torch.tensor([1.,1.,3.,4.],dtype=z.dtype);e=torch.tensor([1,1,0,1])
    grid=torch.tensor([1.,2.,4.],dtype=z.dtype)
    h=breslow_hazard(z,t,e,grid)
    expected=torch.stack([2/z.exp().sum(),2/z.exp().sum(),2/z.exp().sum()+torch.exp(-z[3])])
    torch.testing.assert_close(h,expected)
    s=ph_survival(z,h);weights=quadrature_weights(grid)
    loss=cox_loss(z,t,e)+weighted_brier(s,torch.ones_like(s),torch.zeros_like(s),weights)
    grad=torch.autograd.grad(loss,z)[0]
    assert torch.isfinite(grad).all()
    eps=1e-5;direction=z.new_tensor([.3,-.2,.4,.1])
    def objective(x):return cox_loss(x,t,e)+weighted_brier(ph_survival(x,breslow_hazard(x,t,e,grid)),torch.ones_like(s),torch.zeros_like(s),weights)
    numeric=(objective(z.detach()+eps*direction)-objective(z.detach()-eps*direction))/(2*eps)
    torch.testing.assert_close(grad@direction,numeric,rtol=1e-7,atol=1e-8)


@pytest.mark.parametrize('power',[0,4])
@pytest.mark.parametrize('arm',['E_S','E_eta'])
def test_rank_one_regularizer_matches_direct_multioutput_parameter_gradient(power,arm):
    torch.manual_seed(9)
    model=ClinicalResidualPH(2,{'include_size':True},use_image=False).double()
    with torch.no_grad():model.readout[-1].weight.normal_(std=.1)
    m=mesh();v={'mesh':[m],'q':torch.tensor([[.2,.5]],dtype=torch.float64),
               'im':torch.empty(1,0,dtype=torch.float64),'base':torch.tensor([.3],dtype=torch.float64)}
    baseline=torch.tensor([.1,.3,.7],dtype=torch.float64);weights=quadrature_weights([1.,2.,4.]);rho=.02
    normal,area,R=normal_geometry(m[0],m[1],power=power)
    u=torch.zeros(6,dtype=torch.float64,requires_grad=True)
    shifted=dict(v,mesh=[(m[0]+area.sqrt()*u[:,None]*normal,*m[1:])])
    eta=model.scores(shifted,0,1)[0]
    if arm=='E_S':
        output=torch.exp(-baseline*eta.exp())
        direct=sum(w*metric_quadratic(torch.autograd.grad(value,u,create_graph=True,retain_graph=True)[0],R)
                   for value,w in zip(output,weights))
    else:direct=metric_quadratic(torch.autograd.grad(eta,u,create_graph=True)[0],R)
    parameters=[p for p in model.parameters() if p.requires_grad]
    expected=torch.autograd.grad(rho*direct,parameters,allow_unused=True)
    reg_step(model,v,model.scores(v,0,1).detach(),baseline,arm,rho,{},weights,.1,power)
    for p,g in zip(parameters,expected):
        if g is None:assert p.grad is None or not p.grad.any()
        else:torch.testing.assert_close(p.grad,g,rtol=2e-7,atol=1e-9)


def test_synthetic_ph_training_updates_all_three_branches():
    torch.manual_seed(13)
    model=ClinicalResidualPH(2,{'include_size':True}).double()
    m=mesh();values={'mesh':[m,(m[0]*1.3,*m[1:]),(m[0]*.8,*m[1:])],
                    'q':torch.randn(3,2,dtype=torch.float64),'im':torch.randn(3,512,dtype=torch.float64),
                    'base':torch.tensor([.1,-.2,.3],dtype=torch.float64)}
    t=torch.tensor([1.,3.,4.],dtype=torch.float64);e=torch.tensor([1,0,1]);grid=t
    weights=quadrature_weights(grid);optim=torch.optim.Adam(model.parameters(),lr=.001)
    initial={k:p.detach().clone() for k,p in model.named_parameters()}
    for _ in range(3):
        optim.zero_grad();z=model.scores(values,0,3)
        s=ph_survival(z,breslow_hazard(z,t,e,grid))
        loss=cox_loss(z,t,e)+weighted_brier(s,torch.ones_like(s),torch.zeros_like(s),weights)
        loss=loss+1e-4*sum(p.square().sum() for p in model.parameters() if p.requires_grad)
        loss.backward();optim.step()
        assert torch.isfinite(loss)
    for name in ['geo.readout.0.weight','image.0.weight','readout.2.weight']:
        assert not torch.equal(dict(model.named_parameters())[name],initial[name])
