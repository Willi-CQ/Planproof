"""Search the sheet again using only door examples proposed by two frozen raster models."""
import json
from pathlib import Path
import time

from PIL import Image

from exemplar import find_other,iou

ARCH='b646a61e256b114258e56a80ab57da8cc53ee5f7d73ce2787c39797a6fa80eca'
LEGACY='c9b00cfdbdcaa6c10cb68fe3165aa5ac0ef0ac31bf362e43568bf909ce6eed66'
BASE='c82c584f5374ab0a50ec2b576c1ddae6b6b1f4ce6348d06762db2f82ff0bbbe6'
ADAPTED='981b5f10c80d629ac38228b4bd4a00895fef263f6925945de80360b7cbee4f67'
CONFIG='4301f783ab41bdfefc0df57889adb39a0442787ae5576d687ebab63136d92430'


def other_support(det,other):
    return max((iou(det['bbox'],item['bbox']) for item in other),default=0)


def merge(rows):
    kept=[]
    for row in sorted(rows,key=lambda d:-d['confidence']):
        if other_support(row,kept)<.35:
            kept.append(row)
    return kept


def search(im,first,second,cut):
    agreed=[]
    for row in sorted(first,key=lambda d:-d['confidence']):
        support=other_support(row,second)
        if support>=.35:
            agreed.append(dict(row,agreement=support,origin='two_models'))
    agreed=merge(agreed)
    nearby=[]
    for i,seed in enumerate(agreed[:3]):
        for row in find_other(im,seed['bbox'],cut):
            nearby.append(dict(row,seed_index=i,origin='same_sheet'))
    proposed=list(agreed)
    for row in nearby:
        if (other_support(row,first+second)>=.35 or
                any(q['seed_index']!=row['seed_index'] and iou(row['bbox'],q['bbox'])>=.35 for q in nearby)):
            proposed.append(row)
    return {'agreement':agreed,'automatic_repeat_search':merge(proposed)}


def main():
    from jobs import fetch,digest
    raw1=json.loads(fetch('arch-fitted/predictions.json',ADAPTED).read_bytes())
    raw2=json.loads(fetch('arch-base/predictions.json',BASE).read_bytes())
    one={r['id']:r for r in raw1}
    two={r['id']:r for r in raw2}
    threshold=json.loads(fetch('exemplar-cal/config.json',CONFIG).read_bytes())['threshold']
    output=[]
    start=time.monotonic()
    for packet,sha in [('architect-pages.json',ARCH),('transfer-pages.json',LEGACY)]:
        index=fetch(packet,sha)
        for row in json.loads(index.read_bytes()):
            pid=row['id']
            p=index.parent/row['file']
            assert digest(p.read_bytes())==one[pid]['sha256']==two[pid]['sha256']
            with Image.open(p) as im:
                arms=search(im,one[pid]['arms']['calibrated'],two[pid]['arms']['published_tiled'],threshold)
            output.append(dict(id=pid,sha256=row['sha256'],arms=arms))
    root=Path('/kaggle/working/automatic-repeats')
    root.mkdir()
    path=root/'predictions.json'
    path.write_text(json.dumps(output,allow_nan=False,sort_keys=True)+'\n')
    (root/'receipt.json').write_text(json.dumps(dict(sha256=digest(path.read_bytes()),threshold=threshold,
         original=BASE,adapted=ADAPTED,calibration=CONFIG,seconds=time.monotonic()-start),indent=2)+'\n')


if __name__=='__main__':
    main()
