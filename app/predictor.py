"""Real-time session recommendation adapter for the formal checkpoints.

Given a sequence of raw Diginetica item IDs, build the model-specific session
graph and run ``full_sort_predict`` for the selected model.  The graph building
mirrors ``recbole_gnn.data.dataset`` and the batch transform.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / '03_代码与实验'
REPO = ROOT / 'RecBole-GNN'
sys.path.insert(0, str(REPO))


class SessionPredictor:
    def __init__(self, model_name, dataset='diginetica-formal', seed=42):
        import os
        import torch
        from recbole.utils import init_seed, init_logger
        from recbole_gnn.config import Config
        from recbole_gnn.utils import create_dataset, get_model

        torch.set_num_threads(4)
        os.chdir(ROOT)
        self.model_name = model_name
        self.device = 'cpu'
        result_path = ROOT / 'outputs/formal' / model_name / f'seed{seed}' / 'result.json'
        import json
        self.prior = json.loads(result_path.read_text(encoding='utf-8'))
        self.config = Config(
            model=model_name,
            dataset=dataset,
            config_file_list=[str(ROOT / 'configs/formal.yaml')],
            config_dict={
                'data_path': str(ROOT / 'data'),
                'seed': seed,
                'checkpoint_dir': str(ROOT / 'outputs/formal' / model_name / f'seed{seed}' / 'checkpoints'),
                'use_gpu': False,
            },
        )
        init_seed(self.config['seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.dataset = create_dataset(self.config)
        built = self.dataset.build()
        self.train_dataset = built[0]
        self.model = get_model(model_name)(self.config, self.train_dataset).to(self.device)
        state = torch.load(Path(self.prior['checkpoint']), map_location=self.device)
        self.model.load_state_dict(state['state_dict'])
        self.model.load_other_parameter(state.get('other_parameter'))
        self.model.eval()
        self.item_token2id = self.dataset.field2token_id[self.config['ITEM_ID_FIELD']]
        self.seq_len_field = self.config['ITEM_SEQ_LEN'] or self.config['ITEM_LIST_LENGTH_FIELD']
        self.max_len = int(self.config['MAX_ITEM_LIST_LENGTH'])

    def valid_item(self, token):
        return str(token) in self.item_token2id

    def _bidirectional(self, edge_index):
        import torch
        seq_len = edge_index.shape[1]
        ed = edge_index.T
        ed2 = edge_index.T.flip(dims=[1])
        idc = ed.unsqueeze(1).expand(-1, seq_len, 2) == ed2.unsqueeze(0).expand(seq_len, -1, 2)
        return torch.logical_and(idc[:, :, 0], idc[:, :, 1]).any(dim=-1)

    def _interaction(self, ids):
        import torch
        from recbole.data.interaction import Interaction

        ids = [int(self.item_token2id[str(x)]) for x in ids]
        if self.model_name == 'GCEGNN':
            ids = list(reversed(ids))
            seq = torch.tensor(ids, dtype=torch.long)
            uniq, inverse = torch.unique(seq, return_inverse=True)
            alias = inverse[:len(seq)]
            edge_b = torch.stack([alias[:-1], alias[1:]])
            attr_b = torch.where(self._bidirectional(edge_b), 3, 1)
            edge_f = torch.stack([alias[1:], alias[:-1]])
            attr_f = torch.where(self._bidirectional(edge_f), 3, 2)
            self_loop = torch.stack([alias, alias])
            attr_s = torch.zeros_like(alias)
            edge = torch.cat([
                torch.cat([edge_b, attr_b.unsqueeze(0)], dim=0),
                torch.cat([edge_f, attr_f.unsqueeze(0)], dim=0),
                torch.cat([self_loop, attr_s.unsqueeze(0)], dim=0),
            ], dim=-1).long().unique(dim=-1)
            edge_index, edge_attr = edge[:2], edge[2]
        else:
            seq = torch.tensor(ids, dtype=torch.long)
            uniq, inverse = torch.unique(seq, return_inverse=True)
            alias = inverse[:len(seq)]
            edge_index = torch.stack([alias[:-1], alias[1:]]).unique(dim=-1)
            edge_attr = None

        x = torch.cat([torch.zeros(1, dtype=torch.long), uniq])
        interaction = Interaction({
            'x': x,
            'edge_index': edge_index + 1,
            'alias_inputs': (alias + 1).unsqueeze(0),
            self.seq_len_field: torch.tensor([len(seq)], dtype=torch.long),
            self.config['ITEM_ID_FIELD']: torch.tensor([0], dtype=torch.long),
        })
        if edge_attr is not None:
            interaction['edge_attr'] = edge_attr
        return interaction

    def predict(self, item_ids, topk=10):
        import torch
        item_ids = [str(x) for x in item_ids if str(x).strip()]
        if not item_ids:
            raise ValueError('empty session')
        item_ids = item_ids[-self.max_len:]
        interaction = self._interaction(item_ids)
        with torch.no_grad():
            scores = self.model.full_sort_predict(interaction).reshape(-1, self.dataset.item_num)
        scores = scores.clone()
        scores[:, 0] = -float('inf')
        values, indices = torch.topk(scores, k=topk, dim=-1)
        raw = self.dataset.id2token(self.config['ITEM_ID_FIELD'], indices[0].cpu().numpy()).tolist()
        return [
            {'rank': i + 1, 'item': str(item), 'score': float(values[0, i].item())}
            for i, item in enumerate(raw)
        ]

    def graph(self, item_ids):
        """Return local graph nodes/edges for a browsing-order sequence."""
        import torch
        ids = [str(x) for x in item_ids if str(x).strip()][-self.max_len:]
        seq = torch.tensor([int(self.item_token2id[x]) for x in ids], dtype=torch.long)
        uniq, inverse = torch.unique(seq, return_inverse=True)
        alias = inverse[:len(seq)]
        nodes = []
        for idx, token_id in enumerate(uniq.tolist()):
            count = int((alias == idx).sum().item())
            nodes.append({'id': str(self.dataset.id2token(self.config['ITEM_ID_FIELD'], token_id)), 'count': count})
        edges = []
        for a, b in zip(alias[:-1].tolist(), alias[1:].tolist()):
            key = (a, b)
            existing = next((e for e in edges if e['source_idx'] == a and e['target_idx'] == b), None)
            if existing:
                existing['value'] += 1
            else:
                edges.append({'source_idx': a, 'target_idx': b, 'value': 1})
        for e in edges:
            e['source'] = nodes[e['source_idx']]['id']
            e['target'] = nodes[e['target_idx']]['id']
            e.pop('source_idx', None)
            e.pop('target_idx', None)
        return {'nodes': nodes, 'edges': edges}

    def global_neighbours(self, item_id, limit=12):
        import torch
        if self.model_name != 'GCEGNN':
            return []
        try:
            idx = int(self.item_token2id[str(item_id)])
        except Exception:
            return []
        adj = self.model.global_adj.cpu()
        weight = self.model.global_weight.cpu()
        out = []
        for j in range(adj.shape[1]):
            nid = int(adj[idx, j].item())
            if nid <= 0:
                continue
            token = str(self.dataset.id2token(self.config['ITEM_ID_FIELD'], nid))
            if token == str(item_id) or any(x['id'] == token for x in out):
                continue
            out.append({'id': token, 'weight': float(weight[idx, j].item())})
            if len(out) >= limit:
                break
        return out
