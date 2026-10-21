"""只读取已保存预测，核对分组、验证准确率和最终融合提交。"""
from pathlib import Path
import json,csv
import numpy as np
P=Path(__file__).resolve().parent/'results/ensemble'
def rows(f):
    with f.open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
manifest=rows(P/'fold_manifest.csv')
y=np.array([int(r['target']) for r in manifest]); groups={}
for r in manifest:
    groups.setdefault(r['group'],set()).add(r['fold'])
assert all(len(v)==1 for v in groups.values()),'Duplicate group crosses folds'
metrics=json.loads((P/'validation_metrics.json').read_text(encoding='utf-8'))
preds=[];inds=[];vals=[]
for fold in range(3):
    with np.load(P/f'convnext_fold{fold}.predictions.npz') as a:
        idx=a['indices']; val=a['val'];test=a['test'];preds.append(test);inds.append(idx);vals.append(val)
        assert np.allclose(test.sum(1),1,atol=1e-4)
        acc=float(np.mean(val.argmax(1)==y[idx])); assert abs(acc-metrics['fold_validation_accuracy'][str(fold)])<1e-10
        assert all(int(manifest[i]['fold'])==fold for i in idx)
        print(f'fold{fold}: n={len(idx)}, accuracy={acc:.10f}')
idx=np.concatenate(inds); assert len(idx)==len(set(idx))==16960
assert abs(np.mean(np.concatenate(vals).argmax(1)==y[idx])-metrics['selected_oof_accuracy'])<1e-10
probs=np.mean(preds,axis=0)
sub=rows(P/'submission.csv'); assert len(sub)==7065 and len({r['ID'] for r in sub})==7065
matches={r['ID']:r['Emotion'] for r in json.loads((P/'exact_train_test_matches.json').read_text(encoding='utf-8'))}
labels=probs.argmax(1)
for i,r in enumerate(sub):
    if r['ID'] in matches:
        labels[i]=matches[r['ID']]
        probs[i]=np.eye(6)[int(matches[r['ID']])]
assert np.allclose(probs,np.load(P/'test_probabilities.npy'),atol=1e-7)
assert np.array_equal(labels,np.array([int(r['Emotion']) for r in sub]))
assert len(matches)==511
print('PASS: group isolation, 3-fold validation, probability fusion, 511 exact matches, final CSV')
