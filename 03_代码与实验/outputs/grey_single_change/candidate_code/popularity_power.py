"""A fixed, interpretable grey-relational session recommender (not GM forecasting).

All cross-session statistics come from one longest TRAIN prefix per session.
The current query history is available at prediction time, including repeat clicks.
"""
import numpy as np
from scipy import sparse

FEATURES = ['last_click_transition', 'recency_weighted_cooccurrence',
            'recency_weighted_repeat', 'log_training_popularity']


def grey_coefficients(normalized, rho=0.5):
    """Deng coefficient relative to all-one ideal; extrema over items x features."""
    if not 0 < rho <= 1:
        raise ValueError('rho must be in (0, 1]')
    delta = np.abs(1.0 - normalized)
    lo, hi = float(delta.min()), float(delta.max())
    if hi == 0:
        return np.ones_like(normalized)
    return (lo + rho * hi) / (delta + rho * hi)


def stable_topk(scores, k=10):
    """All items except padding; exact score ties use ascending internal item ID."""
    values = scores[1:]
    k = min(k, len(values))
    boundary = np.partition(values, len(values) - k)[len(values) - k]
    above = np.flatnonzero(values > boundary) + 1
    equal = np.flatnonzero(values == boundary)[:k - len(above)] + 1
    ids = np.concatenate([above, equal])
    return ids[np.lexsort((ids, -scores[ids]))]


def ranking_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    hit = ranks > 0
    reciprocal = np.zeros(len(ranks), dtype=np.float64)
    discounted = reciprocal.copy()
    reciprocal[hit] = 1.0 / ranks[hit]
    discounted[hit] = 1.0 / np.log2(ranks[hit] + 1)
    recall = float(hit.mean())
    return {'recall@10': recall, 'mrr@10': float(reciprocal.mean()),
            'ndcg@10': float(discounted.mean()), 'hit@10': recall,
            'precision@10': recall / 10}


class GreySession:
    def __init__(self, item_num, rho=0.5):
        self.item_num = item_num
        self.rho = rho
        self.weights = np.full(4, 0.25, dtype=np.float32)

    def fit(self, sequences):
        sequences = list(sequences)
        self.counts = np.zeros(self.item_num, dtype=np.int64)
        incidence_rows, incidence_cols, trans_src, trans_dst = [], [], [], []
        for row, seq in enumerate(sequences):
            seq = np.asarray(seq, dtype=np.int64)
            if len(seq) < 1 or np.any(seq <= 0) or np.any(seq >= self.item_num):
                raise ValueError('Invalid training sequence')
            np.add.at(self.counts, seq, 1)
            unique = np.unique(seq)
            incidence_rows.extend([row] * len(unique))
            incidence_cols.extend(unique.tolist())
            trans_src.extend(seq[:-1].tolist())
            trans_dst.extend(seq[1:].tolist())
        incidence = sparse.csr_matrix(
            (np.ones(len(incidence_rows), dtype=np.float32),
             (incidence_rows, incidence_cols)), shape=(len(sequences), self.item_num))
        session_counts = np.asarray(incidence.sum(axis=0)).ravel()
        inv = np.zeros(self.item_num, dtype=np.float32)
        np.divide(1.0, np.sqrt(session_counts), out=inv, where=session_counts > 0)
        cooc = (incidence.T @ incidence).tocsr()
        cooc.setdiag(0)
        cooc.eliminate_zeros()
        self.cooccurrence = (sparse.diags(inv) @ cooc @ sparse.diags(inv)).tocsr()
        transitions = sparse.csr_matrix(
            (np.ones(len(trans_src), dtype=np.float32), (trans_src, trans_dst)),
            shape=(self.item_num, self.item_num))
        outgoing = np.asarray(transitions.sum(axis=1)).ravel()
        inv = np.zeros(self.item_num, dtype=np.float32)
        np.divide(1.0, outgoing, out=inv, where=outgoing > 0)
        self.transitions = (sparse.diags(inv) @ transitions).tocsr()
        self.log_popularity = np.log1p(self.counts).astype(np.float32)
        return self

    def normalized_features(self, history):
        history = np.asarray(history, dtype=np.int64)
        if len(history) == 0 or np.any(history <= 0) or np.any(history >= self.item_num):
            raise ValueError('Query must contain valid non-padding item IDs')
        x = np.zeros((4, self.item_num), dtype=np.float32)
        x[0] = self.transitions.getrow(history[-1]).toarray().ravel()
        # Linear recency weights: 1, 2, ..., L, divided by their sum.
        recency = np.arange(1, len(history) + 1, dtype=np.float32)
        recency /= recency.sum()
        for item, weight in zip(history, recency):
            start, end = self.cooccurrence.indptr[item:item + 2]
            x[1, self.cooccurrence.indices[start:end]] += weight * self.cooccurrence.data[start:end]
        np.add.at(x[2], history, recency)
        x[3] = self.log_popularity
        # Benefit-type min-max scaling across the FULL candidate catalogue.
        # A constant column is mapped to zero and contributes the same to all items.
        candidate = x[:, 1:]
        lo = candidate.min(axis=1, keepdims=True)
        span = candidate.max(axis=1, keepdims=True) - lo
        np.subtract(candidate, lo, out=candidate)
        np.divide(candidate, span, out=candidate, where=span > 0)
        candidate[span[:, 0] == 0] = 0
        candidate[3] **= 0.0625
        return x

    def recommend(self, history):
        x = self.normalized_features(history)
        coefficients = grey_coefficients(x[:, 1:], self.rho)
        scores = np.full(self.item_num, -np.inf, dtype=np.float32)
        scores[1:] = np.sum(self.weights[:, None] * coefficients, axis=0)
        return stable_topk(scores), scores, x, coefficients

    def save(self, path):
        payload = {'counts': self.counts, 'rho': np.asarray(self.rho),
                   'item_num': np.asarray(self.item_num), 'weights': self.weights,
                   'log_popularity': self.log_popularity}
        for name in ['transitions', 'cooccurrence']:
            matrix = getattr(self, name)
            for key in ['data', 'indices', 'indptr']:
                payload[name + '_' + key] = getattr(matrix, key)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(int(data['item_num']), float(data['rho']))
            for name in ['counts', 'weights', 'log_popularity']:
                setattr(model, name, data[name])
            for name in ['transitions', 'cooccurrence']:
                setattr(model, name, sparse.csr_matrix(
                    tuple(data[name + '_' + k] for k in ['data', 'indices', 'indptr']),
                    shape=(model.item_num, model.item_num)))
        return model


def self_check():
    # Analytic coefficient values and ranking metric identities.
    np.testing.assert_allclose(grey_coefficients(np.array([[0., .5, 1.]])),
                               [[1 / 3, .5, 1.]])
    metrics = ranking_metrics([1, 2, 0])
    np.testing.assert_allclose([metrics['recall@10'], metrics['mrr@10']], [2 / 3, .5])
    assert stable_topk(np.array([-np.inf, .5, .8, .8]), 2).tolist() == [2, 3]
    toy = GreySession(7).fit([[1, 2, 1], [2, 3]])
    assert toy.counts.tolist() == [0, 2, 2, 1, 0, 0, 0]
    assert toy.transitions[1, 2] == 1 and toy.cooccurrence[1, 3] == 0
    # Held-out ID 6 may enter a QUERY, but cannot enter training statistics.
    toy.recommend([1, 6])
    assert toy.counts[6] == 0 and toy.transitions.getrow(6).nnz == 0
    assert toy.cooccurrence.getrow(6).nnz == 0
    return True
