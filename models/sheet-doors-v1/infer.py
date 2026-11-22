"""Infer single swing-door boxes on an unmarked paper drawing in original pixel coordinates."""
import argparse
import json
from pathlib import Path

from PIL import Image
from detection_boxes import image_array, merge, positions

PROFILES = {'normal': (1050,), 'large': (1750,), 'multiscale': (1050,1750,2450)}


def candidates(model, image, long_side, threshold, view=None):
    if view is None:
        from detector import view_predictions as view
    original = Image.fromarray(image_array(image))
    w,h = original.size
    scale = long_side/max(w,h)
    sw,sh = max(1,round(w*scale)),max(1,round(h*scale))
    sheet=original.resize((sw,sh),Image.Resampling.BILINEAR)
    boxes=[]
    for y in positions(sh,512,.25):
        for x in positions(sw,512,.25):
            region=sheet.crop((x,y,min(x+512,sw),min(y+512,sh)))
            for row in view(model,region,threshold):
                if row['category']!='single_door':
                    continue
                left,top,right,bottom=row['bbox']
                if ((x>0 and left<3) or (y>0 and top<3) or
                        (x+512<sw and right>509) or (y+512<sh and bottom>509)):
                    continue
                boxes.append(dict(category='single_door',confidence=row['confidence'],
                                  bbox=[(x+left)*w/sw,(y+top)*h/sh,(x+right)*w/sw,(y+bottom)*h/sh]))
    return merge(boxes)


def run(model,image,profile,cut,view=None):
    return merge([d for size in PROFILES[profile] for d in candidates(model,image,size,cut,view)])


if __name__=='__main__':
    import torch
    from detector import load_model
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('weights',type=Path)
    p.add_argument('image',type=Path)
    args=p.parse_args()
    config=json.loads(args.weights.with_name('sheet-config.json').read_text())
    import hashlib
    if hashlib.sha256(args.weights.read_bytes()).hexdigest()!=config['weights_sha256']:
        raise ValueError('Unexpected weights for this configuration')
    torch.set_num_threads(3)
    model,threshold=load_model(args.weights)
    with Image.open(args.image) as image:
        print(json.dumps(run(model,image,config['profile'],threshold),allow_nan=False))
