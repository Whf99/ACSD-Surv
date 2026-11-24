"""Full-risk-set PH calculations with Breslow ties and training-only IPCW."""
import torch


def horizon_support(times, events, grid):
    """Training/validation feasibility only; never select a grid on test labels."""
    cases=((times[:, None]<=grid[None, :]) & events.bool()[:, None]).sum(0)
    controls=(times[:, None]>grid[None, :]).sum(0)
    if bool((cases==0).any()) or bool((controls==0).any()):
        raise ValueError('Prespecified horizon lacks event or control support in training/validation')
    return {'cases':cases.tolist(),'controls':controls.tolist()}


def breslow_hazard(scores, times, events, grid):
    unique = torch.unique(times[events.bool()], sorted=True)
    if not len(unique):
        raise ValueError('Training split contains no events')
    risk = times[None, :] >= unique[:, None]
    counts = ((times[None, :] == unique[:, None]) & events.bool()[None, :]).sum(1)
    # logsumexp avoids overflow in the risk-set denominator.
    log_denom = torch.logsumexp(scores[None, :].expand_as(risk).masked_fill(~risk, -torch.inf), 1)
    increments = counts.to(scores) * torch.exp(-log_denom)
    return increments @ (unique[:, None] <= grid[None, :]).to(scores)


def cox_loss(scores, times, events):
    if not bool(events.bool().any()):
        raise ValueError('Training split contains no events')
    risk = times[None, :] >= times[:, None]
    log_denom = torch.logsumexp(scores[None, :].expand_as(risk).masked_fill(~risk, -torch.inf), 1)
    return (log_denom - scores)[events.bool()].mean()


def ph_survival(scores, baseline):
    return torch.exp(-scores.exp()[:, None] * baseline[None, :])


def weighted_brier(survival, alive_weights, dead_weights, weights):
    return (weights * (alive_weights * (1-survival).square()
                       + dead_weights * survival.square()).mean(0)).sum()
