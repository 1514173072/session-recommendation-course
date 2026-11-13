"""Sparse exact-ranking GRA with context transitions and session-neighbour evidence.

This is a project-designed multiattribute ranker, not a published GRA algorithm.
The support reduction is exact: outside the union all local features are zero;
their score is monotone in static popularity. Keeping its first K items and
the first K internal IDs covers popularity and zero-weight tie cases.
"""
import json
from pathlib import Path
import numpy as np
from scipy import sparse

from grey_session import GreySession, grey_coefficients

FEATURES = ['last_transition', 'history_cooccurrence', 'repeat_recency',
            'log_popularity', 'session_neighbours', 'history_transition']


def top_ids(ids, scores, k=10):
    k = min(k, len(ids))
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)
    # ids are already ascending; ties must match dense catalogue ranking.
    selected = np.concatenate((above, tied[:k - len(above)]))
    order = np.lexsort((ids[selected], -scores[selected]))
    return selected[order]


class FastGreySession:
    def __init__(self, original, incidence, session_users, neighbours=100):
        self.original = original
        self.item_num = original.item_num
        self.incidence = incidence.tocsr()
        self.postings = self.incidence.T.tocsr()
        self.session_users = np.asarray(session_users, dtype=np.int64)
        self.user_row = {int(u): i for i, u in enumerate(self.session_users)}
        self.session_length = np.diff(self.incidence.indptr).astype(np.float32)
        self.neighbours = neighbours
        self.rho = .5
        self.power = 1.0
        self.weights = np.full(6, 1 / 6, dtype=np.float32)
        self.popularity = original.log_popularity
        pop_ids = np.arange(1, self.item_num)
        self.pop_order = pop_ids[np.lexsort((pop_ids, -self.popularity[1:]))]
        self.pop_min = float(self.popularity[1:].min())
        self.pop_max = float(self.popularity[1:].max())

    def _row(self, matrix, item):
        a, b = matrix.indptr[item:item + 2]
        return matrix.indices[a:b], matrix.data[a:b]

    def neighbour_scores(self, history, recency, user=None):
        parts, weights = [], []
        for item, weight in zip(history, recency):
            sessions, _ = self._row(self.postings, item)
            parts.append(sessions)
            weights.append(np.full(len(sessions), weight, dtype=np.float32))
        if not sum(map(len, parts)):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        rows = np.concatenate(parts)
        raw = np.concatenate(weights)
        unique, reverse = np.unique(rows, return_inverse=True)
        similarity = np.bincount(reverse, weights=raw).astype(np.float32)
        # Cosine with position-weighted query; query norm is constant across neighbours.
        similarity /= np.sqrt(self.session_length[unique])
        if user is not None and int(user) in self.user_row:
            similarity[unique == self.user_row[int(user)]] = 0
        positive = similarity > 0
        unique, similarity = unique[positive], similarity[positive]
        if len(unique) > self.neighbours:
            selected = top_ids(unique, similarity, self.neighbours)
            unique, similarity = unique[selected], similarity[selected]
        if not len(unique):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        items, values = [], []
        for row, value in zip(unique, similarity):
            a, b = self.incidence.indptr[row:row + 2]
            items.append(self.incidence.indices[a:b])
            values.append(np.full(b - a, value * value, dtype=np.float32))
        items = np.concatenate(items)
        unique_items, reverse = np.unique(items, return_inverse=True)
        scores = np.bincount(reverse, weights=np.concatenate(values)).astype(np.float32)
        return unique_items, scores

    def features(self, history, user=None, dense=False):
        history = np.asarray(history, dtype=np.int64)
        if len(history) == 0 or np.any(history <= 0) or np.any(history >= self.item_num):
            raise ValueError('Invalid non-padding query history')
        position = np.arange(1, len(history) + 1, dtype=np.float32)
        position /= position.sum()
        ti, tv = self._row(self.original.transitions, history[-1])
        ci, cv, hi, hv = [], [], [], []
        for item, weight in zip(history, position):
            ids, values = self._row(self.original.cooccurrence, item)
            ci.append(ids)
            cv.append(values * weight)
            ids, values = self._row(self.original.transitions, item)
            hi.append(ids)
            hv.append(values * weight)
        ci, cv = np.concatenate(ci), np.concatenate(cv)
        hi, hv = np.concatenate(hi), np.concatenate(hv)
        ni, nv = self.neighbour_scores(history, position, user)
        # Two fallbacks preserve exact Top-10, even if popularity weight is zero.
        support = np.unique(np.concatenate((ti, ci, hi, history, ni,
                                            self.pop_order[:10], np.arange(1, min(11, self.item_num)))))
        if dense:
            support = np.arange(1, self.item_num)
        x = np.zeros((len(support), 6), dtype=np.float32)
        for col, ids, values in [(0, ti, tv), (1, ci, cv), (2, history, position),
                                  (4, ni, nv), (5, hi, hv)]:
            np.add.at(x[:, col], np.searchsorted(support, ids), values)
        x[:, 3] = self.popularity[support]
        # Missing support entries equal zero in the FULL catalogue.
        lo = np.zeros(6, dtype=np.float32)
        if len(support) == self.item_num - 1:
            lo = x.min(axis=0)
        lo[3] = self.pop_min
        maximum = x.max(axis=0)
        maximum[3] = self.pop_max
        span = maximum - lo
        x -= lo
        np.divide(x, span, out=x, where=span > 0)
        x[:, span == 0] = 0
        return support, x

    def coefficients(self, x, rho=None, power=None):
        rho = self.rho if rho is None else rho
        power = self.power if power is None else power
        # A positive repeat feature and its zero values guarantee global Δmin=0, Δmax=1.
        return rho / (1.0 - np.power(x, power) + rho)

    def predict(self, history, user=None, dense=False, return_detail=False):
        ids, features = self.features(history, user, dense=dense)
        coefficients = self.coefficients(features)
        scores = np.sum(coefficients * self.weights[None, :], axis=1)
        chosen = top_ids(ids, scores)
        if return_detail:
            return ids[chosen], scores[chosen], {'support': len(ids), 'features': features[chosen],
                                               'coefficients': coefficients[chosen]}
        return ids[chosen], scores[chosen]

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.original.save(folder / 'statistics.npz')
        sparse.save_npz(folder / 'session_incidence.npz', self.incidence)
        np.save(folder / 'session_users.npy', self.session_users, allow_pickle=False)
        config = {'rho': self.rho, 'power': self.power, 'weights': self.weights.tolist(),
                  'neighbours': self.neighbours, 'features': FEATURES}
        (folder / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        model = cls(GreySession.load(folder / 'statistics.npz'),
                    sparse.load_npz(folder / 'session_incidence.npz'),
                    np.load(folder / 'session_users.npy', allow_pickle=False), config['neighbours'])
        model.rho, model.power = config['rho'], config['power']
        model.weights = np.asarray(config['weights'], dtype=np.float32)
        return model

    def array_bytes(self):
        matrices = [self.incidence, self.postings, self.original.cooccurrence, self.original.transitions]
        arrays = [self.original.counts, self.original.log_popularity, self.session_users,
                  self.session_length, self.pop_order, self.weights]
        return sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in matrices) + sum(a.nbytes for a in arrays)
