"""Survival discrimination and probability evaluation on a fixed time grid.

No test-driven horizon selection; no patient deletion. Time-dependent metrics
explicitly administratively censor beyond the horizon. C-index uses original
labels. IPCW uses training censoring KM only. sksurv's mean_auc uses the test
EVENT survival distribution for integration, not a test censoring model.
"""
import hashlib, json
from pathlib import Path
import numpy as np
import sksurv
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
from sksurv.nonparametric import CensoringDistributionEstimator

DEFAULT_GRID=np.arange(12.,37.)

def labels(time,event):
    t=np.asarray(time,dtype=float); e=np.asarray(event)
    if t.ndim!=1 or e.shape!=t.shape or not np.isfinite(t).all() or (t<=0).any(): raise ValueError('Require finite positive 1-D durations and matching event indicators')
    if not np.isin(e,[0,1,False,True]).all(): raise ValueError('Event must be 0/1')
    return Surv.from_arrays(e.astype(bool),t)

def administrative_censor(time,event,tau):
    t=np.asarray(time,dtype=float); e=np.asarray(event,dtype=bool)
    admin=float(np.nextafter(tau,np.inf)); changed=t>admin
    return labels(np.minimum(t,admin),e&~changed),dict(tau_months=float(tau),administrative_time=admin,retained_n=len(t),truncated_n=int(changed.sum()),later_events_censored_n=int((changed&e).sum()))

def align_curves(curves,native_grid,common_grid,n):
    native=np.asarray(native_grid,dtype=float); common=np.asarray(common_grid,dtype=float); s=np.asarray(curves,dtype=float)
    if native.ndim!=1 or len(native)<2 or (np.diff(native)<=0).any(): raise ValueError('native_grid must be strictly increasing with at least 2 points')
    if s.shape!=(n,len(native)): raise ValueError('curves must have shape [patients, native_grid]')
    if not np.isfinite(s).all() or ((s<0)|(s>1)).any() or (np.diff(s,axis=1)>1e-6).any(): raise ValueError('Invalid survival probabilities or increasing survival curve')
    if native[0]>common[0] or native[-1]<common[-1]: raise ValueError('Native predictions do not cover frozen evaluation horizon; extrapolation prohibited')
    # Shared explicit representation: native survival samples form a
    # right-continuous step function. Predict directly on common_grid if possible.
    return s[:,np.searchsorted(native,common,side='right')-1]

def evaluate_fold(train_time,train_event,test_time,test_event,risks,curves,native_grid,common_grid=DEFAULT_GRID,positivity=.05,administrative=True):
    train=labels(train_time,train_event); test=labels(test_time,test_event)
    grid=np.asarray(common_grid,dtype=float)
    if grid.ndim!=1 or len(grid)<2 or not np.isfinite(grid).all() or (np.diff(grid)<=0).any(): raise ValueError('Common grid must be finite, strictly increasing and length>=2')
    risk=np.asarray(risks,dtype=float) if risks is not None else None
    result=dict(Cindex=None,IAUC=None,IBS=None,reasons={},n_train=len(train),n_test=len(test),time_unit='months',common_grid=grid.tolist(),sksurv_version=sksurv.__version__,ipcw_source='training censoring Kaplan-Meier',auc_weighting='sksurv cumulative_dynamic_auc returned mean_auc, test event survival distribution',curve_alignment='right-continuous previous native value',time_auc=[])
    try: common=align_curves(curves,native_grid,grid,len(test))
    except (ValueError,TypeError) as exc:
        common=None; result['reasons'].update(IAUC=str(exc),IBS=str(exc))
    if risk is None and common is not None: risk=-np.trapz(common,grid,axis=1)
    if risk is None or risk.shape!=(len(test),) or not np.isfinite(risk).all(): result['reasons']['Cindex']='Missing or invalid risk scores'
    else:
        try: result['Cindex']=float(concordance_index_censored(test['event'],test['time'],risk)[0])
        except ValueError as exc: result['reasons']['Cindex']=str(exc)
    if common is None: return result
    if administrative:
        eval_test,info=administrative_censor(test_time,test_event,grid[-1]); result['administrative_censoring']=info
    else:
        eval_test=test; result['administrative_censoring']=dict(enabled=False,retained_n=len(test))
    try:
        estimator=CensoringDistributionEstimator().fit(train)
        g=estimator.predict_proba(np.r_[grid,np.nextafter(grid[-1],np.inf)])
        if not np.isfinite(g).all() or g.min()<positivity: raise ValueError('Training censoring positivity below '+str(positivity))
        result['min_training_censor_survival']=float(g.min())
        if grid[0]<eval_test['time'].min() or grid[-1]>=eval_test['time'].max(): raise ValueError('Frozen grid outside test follow-up support; no test-driven grid change performed')
    except ValueError as exc:
        result['reasons'].update(IAUC=str(exc),IBS=str(exc)); return result
    try:
        b=float(integrated_brier_score(train,eval_test,common,grid))
        if not np.isfinite(b): raise ValueError('Nonfinite integrated Brier score')
        result['IBS']=b
    except ValueError as exc: result['reasons']['IBS']=str(exc)
    cases=(eval_test['time'][:,None]<=grid)&eval_test['event'][:,None]
    controls=eval_test['time'][:,None]>grid
    if np.any(cases.sum(0)==0) or np.any(controls.sum(0)==0):
        result['reasons']['IAUC']='At least one frozen time has no cumulative case or no dynamic control; IAUC unavailable without dropping times'
    else:
        try:
            auc,mean_auc=cumulative_dynamic_auc(train,eval_test,1-common,grid)
            if not np.isfinite(auc).all() or not np.isfinite(mean_auc): raise ValueError('Nonfinite time-dependent AUC')
            result['IAUC']=float(mean_auc)
            result['time_auc']=[dict(month=float(t),auc=float(a),cases=int(c),controls=int(d)) for t,a,c,d in zip(grid,auc,cases.sum(0),controls.sum(0))]
        except ValueError as exc: result['reasons']['IAUC']=str(exc)
    return result

def summarize_folds(fold_results):
    result={}
    if len(fold_results)!=5: raise ValueError('Five folds required for official summary')
    for metric in ('Cindex','IAUC','IBS'):
        valid=[r[metric] for r in fold_results if r.get(metric) is not None and np.isfinite(r[metric])]
        result[metric]=dict(mean=float(np.mean(valid)) if len(valid)==5 else None,sd=float(np.std(valid,ddof=1)) if len(valid)==5 else None,valid_folds=len(valid),reason=None if len(valid)==5 else 'At least one fold unavailable; no partial-fold official mean')
    return result

