"""Sobolev sensitivity penalties and contour perturbations."""
import numpy as np
import torch
from acsd_surv.geometry import normal_geometry, geometry_state, metric_quadratic

def piece_scores(model,v,i,vertices):
    m=v['mesh'][i]
    one={'mesh':[(vertices,*m[1:])]}
    for key in ('q','im','base'):one[key]=v[key][i:i+1]
    return model.scores(one,0,1)[0]

def reg_step(model,tr,z,baseline,arm,rho,cache,weights,ell,power):
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    values=[]; alphas=[]
    for i,m in enumerate(tr['mesh']):
        v=m[0].double()
        if i not in cache:cache[i]=normal_geometry(v,m[1].long(),ell,power)
        normals,area,metric=cache[i]
        u=torch.zeros(len(v),dtype=v.dtype,requires_grad=True)
        eta=piece_scores(model,tr,i,v+u[:,None]*area.sqrt()*normals)
        if not torch.allclose(eta.detach(),z[i],atol=1e-5,rtol=1e-6):raise ValueError('Shape replay mismatch')
        gradient=torch.autograd.grad(eta,u,create_graph=True)[0]
        Q=metric_quadratic(gradient,metric)
        h=baseline.detach()*eta.exp()
        alpha=(weights.to(h)*(h*torch.exp(-h)).square()).sum()
        penalty=alpha*Q if arm=='E_S' else Q
        deriv=torch.autograd.grad(penalty,list(params.values()),allow_unused=True)
        for (_,p),g in zip(params.items(),deriv):
            if g is not None:
                if p.grad is None:p.grad=torch.zeros_like(p)
                p.grad.add_(g.detach(),alpha=rho/len(z))
        values.append(float(Q.detach()));alphas.append(float(alpha.detach()))
    return dict(rho=rho,alpha_mean=float(np.mean(alphas)),Q_mean=float(np.mean(values)))

@torch.no_grad()
def perturbation(model,v,months,baseline,dest,prefix='',weights=None):
    import json
    if weights is None:
        from acsd_surv.config import quadrature_weights
        weights=quadrature_weights(months)
    eta=torch.stack([model.scores(v,i,i+1)[0] for i in range(len(v['ids']))])
    sd=float(eta.std());rows=[]
    for i,m in enumerate(v['mesh']):
        vert=m[0].double();state=geometry_state(vert,m[1].long())
        mass=state['mass']/state['area']
        coord=vert[:,2]-state['centroid'][2]
        for label,raw in [('smooth',torch.cos(coord*.075)),('axial',torch.sin(coord*.4))]:
            field=raw/torch.sqrt((mass*raw.square()).sum()).clamp_min(1e-12)
            for sign in (-1,1):
                delta=sign*field
                assert abs(float((mass*delta.square()).sum())-1)<1e-6
                moved=piece_scores(model,v,i,vert+delta[:,None]*state['normals'])
                diff=(torch.exp(-baseline*moved.exp())-torch.exp(-baseline*eta[i].exp())).square()
                absolute=float((moved-eta[i]).abs())
                rows.append(dict(id=v['ids'][i],field=label,sign=sign,S=float(torch.sqrt((weights.to(diff)*diff).sum())),eta=absolute,relative_eta=absolute/sd if sd>1e-10 else None))
    (dest/(prefix+'perturbation.json')).write_text(json.dumps(dict(mm_rms=1,eta_sd=sd,rows=rows),allow_nan=False))
