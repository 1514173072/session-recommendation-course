"""Verify the downloaded ZIP and extract only the two requested original CSVs."""
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
ARCHIVE = RAW / 'amazon-m2-v1.zip'
SELECTED = ['products_train.csv', 'sessions_train.csv']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if ARCHIVE.stat().st_size != 417556865:
        raise ValueError('Unexpected archive size for the selected release')
    with zipfile.ZipFile(ARCHIVE) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError('ZIP CRC failure: ' + bad)
        print('Archive CRC check passed', flush=True)
        entries = [{'name': i.filename, 'bytes': i.file_size, 'compressed_bytes': i.compress_size,
                    'crc32': format(i.CRC,'08x')} for i in z.infolist()]
        selected = []
        for name in SELECTED:
            info = z.getinfo(name)
            path = RAW / name
            if path.exists():
                raise RuntimeError('Output exists; do not silently overwrite: '+name)
            with z.open(info) as src, path.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=8*1024*1024)
            assert path.stat().st_size == info.file_size
            selected.append({'name': name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                row = next(reader)
                if name == 'sessions_train.csv':
                    print(json.dumps({'file':name,'columns':reader.fieldnames,'first_row':row},ensure_ascii=False),flush=True)
                else:
                    print(json.dumps({'file':name,'columns':reader.fieldnames,
                                      'first_product':{k:v[:180] for k,v in row.items()}},ensure_ascii=False),flush=True)
    manifest = {
        'original_source': 'https://kddcup23.github.io/',
        'mirror': 'https://www.kaggle.com/datasets/marquis03/amazon-m2',
        'download_url': 'https://www.kaggle.com/api/v1/datasets/download/marquis03/amazon-m2?datasetVersionNumber=1',
        'mirror_version': 1,
        'archive': {'name': ARCHIVE.name, 'bytes': ARCHIVE.stat().st_size,
                    'sha256': digest(ARCHIVE), 'all_entries_crc_passed': True},
        'archive_entries': entries,
        'archive_uncompressed_bytes': sum(e['bytes'] for e in entries),
        'extracted_files': selected,
        'extracted_csv_bytes': sum(e['bytes'] for e in selected),
        'retained_archive_and_selected_csv_bytes': ARCHIVE.stat().st_size+sum(e['bytes'] for e in selected),
        'data_scope': 'Unmodified original products_train.csv and sessions_train.csv across all six locales; no training or sampling performed.'
    }
    (ROOT / '下载清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['archive_entries','download_url']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
