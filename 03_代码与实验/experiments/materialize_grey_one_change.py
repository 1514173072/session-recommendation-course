"""Create literal one-location patches to original code for every frozen variant."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/grey_single_change'


def materialize():
    source=Path(__file__).with_name('grey_session.py')
    text=source.read_text(encoding='utf-8')
    frozen=json.loads((OUT/'frozen_selection.json').read_text(encoding='utf-8'))
    folder=OUT/'candidate_code'
    folder.mkdir(exist_ok=True)
    manifest={}
    for record in [frozen['baseline']]+frozen['selected_per_family']:
        config=record['config']
        family,value=config['family'],config['value']
        modified=text
        old,new=None,None
        combine='        scores[1:] = np.sum(self.weights[:, None] * coefficients, axis=0)\n'
        weights={'transition_weight':0,'cooccurrence_weight':1,'repeat_weight':2,'popularity_weight':3}
        powers={'transition_power':0,'cooccurrence_power':1,'repeat_power':2,'popularity_power':3}
        if family in weights:
            multiplier=[1.,1.,1.,1.]
            multiplier[weights[family]]=value
            old=combine
            new=f'        scores[1:] = np.sum((self.weights * np.array({multiplier!r}, dtype=np.float32))[:, None] * coefficients, axis=0)\n'
        elif family=='rho':
            old='        coefficients = grey_coefficients(x[:, 1:], self.rho)\n'
            new=f'        coefficients = grey_coefficients(x[:, 1:], {value!r})\n'
        elif family=='all_power':
            old='        return x\n'
            new=f'        candidate **= {value!r}\n'+old
        elif family in powers:
            old='        return x\n'
            new=f'        candidate[{powers[family]}] **= {value!r}\n'+old
        elif family=='all_binary':
            old='        return x\n'
            new='        candidate[:] = candidate > 0\n'+old
        elif family=='repeat_binary':
            old='        np.add.at(x[2], history, recency)\n'
            new='        x[2, history] = 1.0\n'
        elif family=='repeat_count':
            old='        np.add.at(x[2], history, recency)\n'
            new='        np.add.at(x[2], history, 1.0)\n'
        elif family=='repeat_last_only':
            old='        np.add.at(x[2], history, recency)\n'
            new='        x[2, history[-1]] = 1.0\n'
        elif family=='keep_self_cooccurrence':
            old='        cooc.setdiag(0)\n'
            new=''
        elif family=='geometric_mean':
            old=combine
            new='        scores[1:] = np.exp(np.mean(np.log(coefficients), axis=0))\n'
        elif family=='harmonic_mean':
            old=combine
            new='        scores[1:] = 1.0 / np.mean(1.0 / coefficients, axis=0)\n'
        elif family=='max_aggregation':
            old=combine
            new='        scores[1:] = np.max(coefficients, axis=0)\n'
        elif family!='baseline':
            raise ValueError(family)
        if old is not None:
            assert modified.count(old)==1,(family,old)
            modified=modified.replace(old,new,1)
        path=folder/(family+'.py')
        path.write_text(modified,encoding='utf-8')
        lines=list(difflib.unified_diff(text.splitlines(True),modified.splitlines(True),
                                      fromfile='grey_session.py',tofile='grey_session.py'))
        (folder/(family+'.patch')).write_text(''.join(lines),encoding='utf-8')
        added=[l for l in lines if l.startswith('+') and not l.startswith('+++')]
        deleted=[l for l in lines if l.startswith('-') and not l.startswith('---')]
        assert max(len(added),len(deleted))==(0 if family=='baseline' else 1)
        manifest[family]={'config':config,'source_file':str(path),'added_lines':len(added),
                          'deleted_lines':len(deleted),'changed_locations':0 if family=='baseline' else 1,
                          'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT/'single_location_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


if __name__=='__main__':
    print(json.dumps(materialize(),indent=2))
