"""Download official RecBole Diginetica interactions; make a real-session smoke subset."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import random
import shutil
import urllib.request
import urllib.error
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/DIGINETICA/not_merged/diginetica.zip'
FALLBACK_URL = 'https://drive.usercontent.google.com/download?id=1yQtXcZN_VleaZqCCO55I45FBMjwT6hhh&export=download&confirm=t'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=int, default=2000)
    args = parser.parse_args()
    source_url = URL
    archive = ROOT / 'cache/diginetica-official.zip'
    full = ROOT / 'data/diginetica-not-merged/diginetica-not-merged.inter'
    sample = ROOT / 'data/diginetica-smoke/diginetica-smoke.inter'
    for directory in [archive.parent, full.parent, sample.parent, ROOT / 'outputs']:
        directory.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print('Downloading official dataset...', flush=True)
        partial = archive.with_suffix('.zip.partial')
        try:
            response = urllib.request.urlopen(URL, timeout=90)
        except urllib.error.HTTPError as e:
            if e.code != 403:
                raise
            print('S3 returned 403; using the public Google Drive mirror linked by RecSysDatasets.', flush=True)
            source_url = FALLBACK_URL
            response = urllib.request.urlopen(source_url, timeout=90)
        with response, partial.open('wb') as f:
            shutil.copyfileobj(response, f)
        if not zipfile.is_zipfile(partial):
            raise ValueError('Downloaded response is not a ZIP archive.')
        partial.replace(archive)
        archive.with_suffix('.source.txt').write_text(source_url, encoding='utf-8')
    else:
        source_url = archive.with_suffix('.source.txt').read_text(encoding='utf-8')
    if not full.exists():
        with zipfile.ZipFile(archive) as z:
            members = [n for n in z.namelist() if n.endswith('.inter')]
            if len(members) != 1:
                raise ValueError(f'Expected one .inter file, got {members}')
            print('Extracting ' + members[0], flush=True)
            with z.open(members[0]) as src, full.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    counts = Counter()
    items = set()
    with full.open(encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        fields = [x.split(':')[0] for x in header]
        session_index, item_index = fields.index('session_id'), fields.index('item_id')
        for row in reader:
            counts[row[session_index]] += 1
            items.add(row[item_index])
    eligible = sorted(s for s, n in counts.items() if 5 <= n <= 20)
    chosen = set(random.Random(42).sample(eligible, min(args.sessions, len(eligible))))
    sample_rows, sample_items = 0, set()
    with full.open(encoding='utf-8') as src, sample.open('w', encoding='utf-8', newline='') as dst:
        reader = csv.reader(src, delimiter='\t')
        writer = csv.writer(dst, delimiter='\t', lineterminator='\n')
        writer.writerow(next(reader))
        for row in reader:
            if row[session_index] in chosen:
                writer.writerow(row)
                sample_rows += 1
                sample_items.add(row[item_index])
    report = {
        'primary_url': URL, 'source_url': source_url, 'archive_sha256': sha256(archive),
        'full_file': str(full), 'full_sha256': sha256(full),
        'raw_sessions': len(counts), 'raw_items': len(items), 'raw_interactions': sum(counts.values()),
        'smoke_file': str(sample), 'smoke_sha256': sha256(sample),
        'smoke_sessions': len(chosen), 'smoke_items': len(sample_items), 'smoke_interactions': sample_rows,
        'seed': 42, 'selection': 'Complete real sessions with 5-20 raw clicks; seeded random session sample.',
        'scope': 'Smoke verification only; not the official 5-core benchmark protocol.'
    }
    (ROOT / 'outputs/data_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)

if __name__ == '__main__':
    main()
