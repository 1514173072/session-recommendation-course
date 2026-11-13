"""Static packaging and source integrity checks after actual browser checks."""
from pathlib import Path
import hashlib,json,re,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
output=ROOT/'08扩展对比/推荐性能与耗时对比.html'
html=output.read_text(encoding='utf-8')
data=json.loads(re.search(r'<script id="experiment-data" type="application/json">([\s\S]*?)</script>',html).group(1))
assert len(data['models'])==8 and len(data['records'])==20
assert {r['scope'] for r in data['records']}=={'30','1000','20000'}
assert not any(m['id'].startswith(('mlp','cloud_')) for m in data['models'])
assert not any(m['id'] in ['rules','qwen_sync'] for m in data['models'])
assert any(m['id']=='gra_improved' for m in data['models'])
assert all(r['scope']=='30' for r in data['records'] if r['id'] in ['qwen_plain','qwen_teacher'])
assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=',html)
script=re.search(r'<script>([\s\S]*?)</script>',html).group(1)
assert not any(x in script for x in ['localStorage','sessionStorage','fetch(','XMLHttpRequest','createObjectURL'])
js=HERE/'check.js';js.write_text(script,encoding='utf-8')
subprocess.run([r'D:\node\node.exe','--check',str(js)],check=True,capture_output=True)
manifest=json.loads((HERE/'data_manifest.json').read_text(encoding='utf-8'))
for path,digest in manifest['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
assert output.read_bytes()==(HERE/'dist/index.html').read_bytes()
assert hashlib.sha256(output.read_bytes()).hexdigest()==manifest['sha256']
report={'passed':True,'models':8,'records':20,'scope_counts':{s:sum(r['scope']==s for r in data['records']) for s in ['30','1000','20000']},
        'no_external_assets':True,'no_save_export_or_metric_switch':True,'original_online_qwen_only_30_cases':True,
        'sources_unchanged':True,'javascript_syntax_passed':True,'html_sha256':manifest['sha256']}
(HERE/'delivery_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
