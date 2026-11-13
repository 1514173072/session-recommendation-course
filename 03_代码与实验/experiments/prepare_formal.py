"""Build a disjoint formal-experiment subset from the full Diginetica interactions.

Protocol:
1. Apply item 5-core on the FULL dataset (items with fewer than 5 interactions removed).
2. Keep complete sessions whose filtered length is 5-20.
3. Exclude the 2,000-session smoke subset, then sample 20,000 sessions (seed 2026).

This keeps a globally consistent 5-core item vocabulary; the formal subset is a
random sample of the full 5-core session population, not a sample-then-filter set.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / 'data/diginetica-not-merged/diginetica-not-merged.inter'
SMOKE = ROOT / 'data/diginetica-smoke/diginetica-smoke.inter'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_header(reader):
    header = next(reader)
    fields = [x.split(':')[0] for x in header]
    return fields


def session_ids(path):
    ids = set()
    with path.open(encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        fields = read_header(reader)
        idx = fields.index('session_id')
        for row in reader:
            ids.add(row[idx])
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--name', default='diginetica-formal')
    parser.add_argument('--item-core', type=int, default=5)
    args = parser.parse_args()

    out_dir = ROOT / 'data' / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{args.name}.inter'
    manifest_path = ROOT / 'outputs/formal/data_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    smoke_ids = session_ids(SMOKE)

    # Pass 1: count item frequencies and raw session lengths.
    item_counts = Counter()
    session_lengths = Counter()
    with FULL.open(encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        fields = read_header(reader)
        s_idx, i_idx = fields.index('session_id'), fields.index('item_id')
        for row in reader:
            item_counts[row[i_idx]] += 1
            session_lengths[row[s_idx]] += 1
    keep_items = {item for item, n in item_counts.items() if n >= args.item_core}

    # Pass 2: count each session's interactions after item 5-core.
    kept_lengths = Counter()
    with FULL.open(encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        read_header(reader)
        for row in reader:
            if row[i_idx] in keep_items:
                kept_lengths[row[s_idx]] += 1

    eligible = sorted(
        s for s, n in kept_lengths.items()
        if 5 <= n <= 20 and s not in smoke_ids
    )
    chosen = set(random.Random(args.seed).sample(eligible, min(args.sessions, len(eligible))))

    # Pass 3: write the selected complete sessions, retaining only 5-core items.
    rows = 0
    subset_items = set()
    with FULL.open(encoding='utf-8') as src, out_file.open('w', encoding='utf-8', newline='') as dst:
        reader = csv.reader(src, delimiter='\t')
        writer = csv.writer(dst, delimiter='\t', lineterminator='\n')
        writer.writerow(next(reader))
        for row in reader:
            if row[s_idx] in chosen and row[i_idx] in keep_items:
                writer.writerow(row)
                rows += 1
                subset_items.add(row[i_idx])

    manifest = {
        'source_file': str(FULL),
        'source_sha256': sha256(FULL),
        'source_sessions': len(session_lengths),
        'source_items': len(item_counts),
        'source_interactions': sum(session_lengths.values()),
        'item_core': args.item_core,
        'items_after_global_5core': len(keep_items),
        'excluded_smoke_sessions': len(smoke_ids),
        'eligible_sessions': len(eligible),
        'selected_sessions': len(chosen),
        'selected_items': len(subset_items),
        'selected_interactions': rows,
        'seed': args.seed,
        'selection': 'Global item 5-core on the full data; complete 5-20 click sessions; smoke excluded; seeded random sample.',
        'output_file': str(out_file),
        'output_sha256': sha256(out_file),
        'scope': 'Formal course-project experiment subset; not the official leaderboard split.',
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
