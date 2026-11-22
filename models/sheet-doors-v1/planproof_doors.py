"""Inspect one drawing for single swing doors or tagged openings."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

HERE=Path(__file__).resolve().parent


def raster(image,example,weights,repeat_search=False):
    if example is not None:
        from exemplar import find_other
        config=json.loads((HERE/'example-config.json').read_bytes())
        rows=find_other(image,example,config['threshold'])
        return dict(route='supplied_example',target='appearances_like_the_supplied_door',
                    example=list(example),additional_candidates=rows,proposed_total=1+len(rows))
    from infer import run
    config=json.loads(weights.with_name('sheet-config.json').read_bytes())
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=config['weights_sha256']:
        raise ValueError('Weight file differs from calibration record')
    if os.environ.get('PLANPROOF_COMPACT_CPU') == '1' and not repeat_search:
        from planproof_runtime.detector_cpu import load_model, view_predictions
    else:
        import torch
        from detector import load_model, predict, view_predictions
        torch.set_num_threads(3)
    model,threshold=load_model(weights)
    rows=run(model,image,config['profile'],threshold,view_predictions)
    if repeat_search:
        from automatic import search
        original_path=HERE/'open-symbol.npz'
        if hashlib.sha256(original_path.read_bytes()).hexdigest()!=config['start_sha256']:
            raise ValueError('Original detector differs from training record')
        original,original_cut=load_model(original_path)
        original_rows=[r for r in predict(original,image,original_cut,mode='tiled') if r['category']=='single_door']
        repeat_cut=json.loads((HERE/'example-config.json').read_bytes())['threshold']
        repeated=search(image,rows,original_rows,repeat_cut)
        matches=repeated['automatic_repeat_search']
        return dict(route='automatic_repeat_search',target='single_swing_door',candidates=matches,
                    proposed_total=len(matches),weights_sha256=[config['weights_sha256'],config['start_sha256']],
                    repeat_threshold=repeat_cut,agreed_seed_count=len(repeated['agreement']),
                    unconfirmed_candidates=[] if repeated['agreement'] else rows)
    return dict(route='raster_cnn',target='single_swing_door',candidates=rows,proposed_total=len(rows),
                weights_sha256=config['weights_sha256'],profile=config['profile'],threshold=threshold)


def displayed_points(result,sx,sy):
    def mapbox(b): return [float(v/s) for v,s in zip(b,(sx,sy,sx,sy))]
    if 'example' in result: result['example']=mapbox(result['example'])
    for key in ('candidates','additional_candidates','unconfirmed_candidates'):
        for d in result.get(key,[]): d['bbox']=mapbox(d['bbox'])
    return result


def inspect(path,page_number,example,weights,schedule,repeat_search=False):
    raw_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    common=dict(source_sha256=raw_hash,page=page_number,review_status='requires_review')
    if path.suffix.lower()!='.pdf':
        if page_number!=1 or schedule is not None:
            raise ValueError('Raster images use page 1. A door-mark schedule requires a PDF.')
        with Image.open(path) as im:
            return dict(**common,width=im.width,height=im.height,coordinates='pixels, top-left xyxy',
                        **raster(im,example,weights,repeat_search))
    import pymupdf
    from vector_doors import inspect_page
    with pymupdf.open(path) as doc:
        if not 1<=page_number<=len(doc): raise ValueError('Page outside PDF')
        page=doc[page_number-1]
        width,height=page.rect.width,page.rect.height
        common.update(width=width,height=height,coordinates='displayed PDF points, top-left xyxy')
        if example is None:
            native=inspect_page(page,schedule=schedule)
            if native['swings'] or native['marked_openings'] or native['paired_leaves']:
                component_frames=[]
                if native['marked_openings']:
                    from mark_frames import inspect_page as frames_from_jambs
                    component_frames=frames_from_jambs(page)
                return dict(**common,route='vector_pdf',evidence=native,component_frames=component_frames,
                            proposed_single_swing=len(native['swings']) if native['swings'] else None,
                            observed_door_marks=len(native['marked_openings']))
            common['vector_search']={'swing_status':native['swing_status'],'mark_status':native['mark_status'],
                                     'vector_strokes':native['vector_strokes']}
        zoom=min(2.,2600/max(width,height))
        pix=page.get_pixmap(matrix=pymupdf.Matrix(zoom,zoom),colorspace=pymupdf.csRGB,alpha=False)
        sx,sy=pix.width/width,pix.height/height
        im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
        seed=None if example is None else [v*s for v,s in zip(example,(sx,sy,sx,sy))]
        return dict(**common,**displayed_points(raster(im,seed,weights,repeat_search),sx,sy))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('drawing',type=Path)
    parser.add_argument('--page',type=int,default=1)
    parser.add_argument('--weights',type=Path,default=HERE/'eteks-door.npz')
    methods=parser.add_mutually_exclusive_group()
    methods.add_argument('--example',type=float,nargs=4,metavar=('LEFT','TOP','RIGHT','BOTTOM'))
    methods.add_argument('--repeat-search',action='store_true',help='Use experimental two-model repeat search on raster inputs')
    parser.add_argument('--schedule',type=Path,help='Optional JSON object mapping printed opening marks to descriptions')
    a=parser.parse_args()
    schedule=json.loads(a.schedule.read_text(encoding='utf-8')) if a.schedule else None
    if schedule is not None and (not isinstance(schedule,dict) or not all(isinstance(v,str) for v in schedule.values())):
        parser.error('Schedule must map printed marks to class descriptions')
    if schedule is not None and a.example: parser.error('A PDF mark schedule and an example box select different searches')
    result=inspect(a.drawing,a.page,a.example,a.weights,schedule,a.repeat_search)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__': main()
