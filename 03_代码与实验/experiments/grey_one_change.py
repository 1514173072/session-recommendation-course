"""Original FOUR-feature GRA, with exactly one named intervention per variant.

All defaults reproduce grey_session.py. No session-neighbour index, new
feature, sparse query acceleration, or simultaneous hyperparameter tuning.
"""
import numpy as np
from grey_session import grey_coefficients, stable_topk

DEFAULT = {'family': 'baseline', 'value': None}


def configurations():
    groups = {
        'repeat_weight': [0, .5, 1.5, 2, 3, 4, 6, 8, 16, 32],
        'popularity_weight': [0, .05, .1, .25, .5, .75, 1.5, 2],
        'cooccurrence_weight': [0, .5, 1.5, 2, 3, 4],
        'transition_weight': [0, .5, 1.5, 2, 3, 4],
        'rho': [.1, .25, .75, 1.0],
        'all_power': [.00390625, .0078125, .015625, .03125, .0625, .125, .25, .5, .75, 1.5, 2.0],
        'repeat_power': [.0625, .125, .25, .5, 2.0, 4.0],
        'popularity_power': [.0625, .125, .25, .5, 2.0, 4.0],
        'cooccurrence_power': [.0625, .125, .25, .5, .75, 2.0, 4.0],
        'transition_power': [.0625, .125, .25, .5, 2.0, 4.0],
        'repeat_binary': [True],
        'repeat_count': [True],
        'repeat_last_only': [True],
        'keep_self_cooccurrence': [True],
        'geometric_mean': [True],
        'harmonic_mean': [True],
        'max_aggregation': [True],
        'all_binary': [True],
    }
    return [dict(DEFAULT)] + [{'family':f, 'value':v} for f, values in groups.items() for v in values]


def settings(config):
    family, value = config['family'], config['value']
    weights = np.full(4, .25, dtype=np.float32)
    columns = {'transition_weight':0, 'cooccurrence_weight':1,
               'repeat_weight':2, 'popularity_weight':3}
    if family in columns:
        weights[columns[family]] *= value
    return weights, float(value) if family == 'rho' else .5


def transform(x, config, alternate=None):
    """Input: rows=candidates, columns=original four normalized features."""
    x = np.array(x, copy=True)
    family, value = config['family'], config['value']
    if family == 'all_power':
        np.power(x, value, out=x)
    if family == 'all_binary':
        x[:] = x > 0
    power_col = {'repeat_power':2, 'popularity_power':3,
                 'cooccurrence_power':1, 'transition_power':0}
    if family in power_col:
        col = power_col[family]
        x[:, col] = np.power(x[:, col], value)
    if family == 'repeat_binary':
        x[:, 2] = x[:, 2] > 0
    if family == 'repeat_count':
        x[:, 2] = alternate['repeat_count']
    if family == 'repeat_last_only':
        x[:, 2] = alternate['repeat_last_only']
    if family == 'keep_self_cooccurrence':
        x[:, 1] = alternate['self_cooccurrence']
    return x


def combine(coefficients, config):
    weights, _ = settings(config)
    family = config['family']
    if family == 'geometric_mean':
        return np.exp(np.mean(np.log(coefficients), axis=1))
    if family == 'harmonic_mean':
        return 1.0 / np.mean(1.0 / coefficients, axis=1)
    if family == 'max_aggregation':
        return coefficients.max(axis=1)
    # Transpose to retain the original summation direction over the four features.
    return np.sum(weights[:, None] * coefficients.T, axis=0)


def scores_from_cache(x, config, alternate=None):
    x = transform(x, config, alternate)
    _, rho = settings(config)
    # Every tested transform preserves full-catalogue extrema Δmin=0 and Δmax=1.
    coefficients = rho / (1.0 - x + rho)
    return combine(coefficients, config)


class OneChangePredictor:
    """Dense reference: original full-catalogue feature extraction and Top-10."""
    def __init__(self, model, self_model, config):
        self.model, self.self_model, self.config = model, self_model, dict(config)

    def predict(self, history):
        history = np.asarray(history, dtype=np.int64)
        source = self.self_model if self.config['family']=='keep_self_cooccurrence' else self.model
        x = source.normalized_features(history)[:, 1:].T
        alternate = {}
        if self.config['family']=='repeat_count':
            counts = np.bincount(history, minlength=self.model.item_num)[1:].astype(np.float32)
            counts /= counts.max()
            alternate['repeat_count'] = counts
        if self.config['family']=='repeat_last_only':
            alternate['repeat_last_only'] = (np.arange(1,self.model.item_num)==history[-1]).astype(np.float32)
        # The one-line-deletion model already supplies the modified cooccurrence.
        if self.config['family']=='keep_self_cooccurrence':
            alternate['self_cooccurrence'] = x[:, 1]
        x = transform(x, self.config, alternate)
        _, rho = settings(self.config)
        coefficients = grey_coefficients(x, rho)
        values = combine(coefficients, self.config)
        scores = np.concatenate((np.array([-np.inf],dtype=np.float32), values))
        ids = stable_topk(scores)
        return ids, scores[ids]
