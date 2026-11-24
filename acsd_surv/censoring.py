"""Training-only marginal censoring estimator."""
import numpy as np

class CensorKM:
    """Marginal reverse Kaplan-Meier. Requires marginal independent censoring.

    This intentionally does NOT claim to estimate general conditional G(t|X).
    Event-first tie convention: recurrence weights query G(t-).
    """
    def fit(self, times, events):
        times, events = np.asarray(times, float), np.asarray(events, int)
        if len(times) == 0 or np.any(times <= 0) or not np.isfinite(times).all():
            raise ValueError('Censor model needs non-empty positive finite training times')
        self.times = np.unique(times)
        self.values = []
        p = 1.
        for t in self.times:
            # Reverse KM: observed recurrence leaves the risk set before a
            # censoring event at the same recorded time (event-first ties).
            n = np.sum(times >= t) - np.sum((times == t) & (events == 1))
            d = np.sum((times == t) & (events == 0))
            if d:
                p *= 1. - d / n
            self.values.append(p)
        self.values = np.asarray(self.values)
        return self

    def predict(self, t, left=False):
        t = np.asarray(t, float)
        idx = np.searchsorted(self.times, t, side='left' if left else 'right') - 1
        return np.where(idx >= 0, self.values[np.maximum(idx, 0)], 1.)

    def state(self):
        return {'times': self.times.tolist(), 'values': self.values.tolist(), 'estimator': 'marginal_reverse_KM'}

    @classmethod
    def from_state(cls, state):
        model = cls()
        model.times, model.values = np.asarray(state['times']), np.asarray(state['values'])
        return model


