"""Construct a reusable, source-bound page reading without receiving questions or reference files."""
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import csv
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import unicodedata

import numpy as np
from PIL import Image

from .geometry import assign_rooms, center, structural_mask, visible_sides
from .openings import marked_vector_profiles, polygon_sides
from . import ocr

ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/'models/sheet-doors-v1'
DOOR=re.compile(r'(?:D|DR|DOOR)[-.]?\d{1,5}[A-Z]?',re.I)
ROOM_CODE=re.compile(r'\bR[-.]?\d{1,4}\b',re.I)
ROOM_NAME=re.compile(r'(?:corridor|hall(?:way)?|landing|entrance|entry|stair(?:case|well)?|bed(?:room)?|bath(?:room)?|'
                     r'kitchen|living(?: room)?|dining(?: room)?|office|lounge|study|closet|storage|laundry|wc|'
                     r'flur|treppenraum|schlafzimmer|küche|bad|couloir|chambre|séjour|cuisine)(?:\s?\d+[A-Z]?)?',re.I)
AREA=re.compile(r'(?<![\d.,])(\d+(?:[.,]\d+)?)\s*m\s*(?:2|²)(?!\w)',re.I)


def clean(s):
    return ' '.join(unicodedata.normalize('NFKC',s).split())


def evidence(kind,box,text,identifier):
    return dict(id=identifier,kind=kind,bbox=[float(v) for v in box],text=text)


def native_reads(path,page_number):
    import pymupdf
    from backend.scene_geometry import boundary_regions,BoundaryBudgetExceeded
    from backend.scene_native import visible_lines
    with pymupdf.open(path) as doc:
        if not 1<=page_number<=len(doc):
            raise ValueError('Requested PDF page is outside this document')
        page=doc[page_number-1]
        width,height=page.rect.width,page.rect.height
        zoom=min(2.,2300/max(width,height))
        pix=page.get_pixmap(matrix=pymupdf.Matrix(zoom,zoom),colorspace=pymupdf.csRGB,alpha=False)
        blob=pix.tobytes('png')
        image=Image.open(BytesIO(blob)).convert('RGB')
        proxy=SimpleNamespace(pdf_bytes=path.read_bytes(),page_index=page_number-1,width=pix.width,height=pix.height,png=blob)
        def to_page(b):
            return [float(v*s/1000) for v,s in zip(b,(width,height,width,height))]
        lines=[]
        for line in visible_lines(proxy):
            words=[dict(text=w['text'],bbox=to_page(w['bbox'])) for w in line['words']]
            lines.append(dict(text=line['text'],bbox=to_page(line['bbox']),words=words))
        try:
            rings=[dict(points=[[x*width/1000,y*height/1000] for x,y in v['polygon']],
                        explicit=v['boundary_basis']=='explicit closed PDF path') for v in boundary_regions(proxy)]
        except BoundaryBudgetExceeded:
            rings=[]
        has_large_bitmap=any(pymupdf.Rect(x['bbox']).get_area()>.03*width*height for x in page.get_image_info())
        structure=structural_mask(page,pix.width,pix.height)
    return image,lines,rings,width,height,bool(lines) and not has_large_bitmap,structure


def ocr_read(image,width,height):
    """Tesseract word boxes stay in the displayed drawing frame."""
    if not ocr.available():
        return [],'tesseract unavailable'
    scale=min(2.5,2600/max(image.size))
    sampled=image.resize(tuple(max(1,round(scale*v)) for v in image.size),Image.Resampling.LANCZOS)
    try:
        tsv=ocr.read_tsv(sampled,11,45)
    except subprocess.TimeoutExpired:
        return [],'OCR timed out'
    except OSError:
        return [],'OCR failed'
    lines=defaultdict(list)
    for w in csv.DictReader(tsv.splitlines(),delimiter='\t'):
        if not w.get('text','').strip() or float(w['conf'])<35:
            continue
        x,y,ww,hh=(int(w[k]) for k in ('left','top','width','height'))
        b=[x*width/sampled.width,y*height/sampled.height,(x+ww)*width/sampled.width,(y+hh)*height/sampled.height]
        k=tuple(w[v] for v in ('block_num','par_num','line_num'))
        lines[k].append(dict(text=w['text'].strip(),bbox=b))
    result=[]
    for words in lines.values():
        boxes=[w['bbox'] for w in words]
        bounds=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
        result.append(dict(text=' '.join(w['text'] for w in words),bbox=bounds,words=words))
    return result,'OCR, incomplete inventory'


def label_reads(lines):
    labels=defaultdict(list)
    marks=defaultdict(list)
    for index,line in enumerate(lines):
        text=clean(line['text'])
        provenance=({k:line[k] for k in ('reader','rotation_degrees') if k in line}
                    if line.get('reader')=='tesseract-word-crop' else {})
        code=ROOM_CODE.search(text)
        rooms=[(code.group().upper(),next((w['bbox'] for w in line['words'] if clean(w['text']).upper()==code.group().upper()),line['bbox']))] if code else []
        if not rooms and (ROOM_NAME.fullmatch(text) or
                          (re.fullmatch(r'[A-Z]+(?:_[A-Z]+)+',text) and
                           ROOM_NAME.fullmatch(text.split('_')[-1]))):
            rooms=[(text.upper(),line['bbox'])]
        for name,box in rooms:
            labels[name].append({**evidence('room_label',box,name,f'room-label-{index}'),**provenance})
        for j,word in enumerate(line['words']):
            mark=clean(word['text']).strip('.,;:()[]').upper()
            if DOOR.fullmatch(mark):
                marks[mark].append({**evidence('door_mark',word['bbox'],mark,f'door-{index}-{j}'),**provenance})
    resolved={}
    for name,anchors in labels.items():
        first=anchors[0];b=first['bbox']
        # Some sheets print a room ID above its identical, differently cased type.
        # Repeated labels elsewhere remain ambiguous.
        same_reference=all(abs(center(a['bbox'])[1]-center(b)[1])<=2*max(b[3]-b[1],a['bbox'][3]-a['bbox'][1]) and
                           min(b[2],a['bbox'][2])>max(b[0],a['bbox'][0]) for a in anchors[1:])
        if same_reference:resolved[name]=first
    return resolved,dict(marks)


def build_scene(path,page_number=1):
    path=Path(path)
    if str(MODEL) not in sys.path:
        sys.path.insert(0,str(MODEL))
    from planproof_doors import inspect
    pdf=path.suffix.lower()=='.pdf'
    if pdf:
        image,lines,rings,width,height,complete,structure=native_reads(path,page_number)
    else:
        if page_number!=1:
            raise ValueError('Images have one page')
        with Image.open(path) as loaded:
            image=loaded.convert('RGBA')
            paper=Image.new('RGBA',image.size,'white')
            paper.alpha_composite(image)
            image=paper.convert('RGB')
        width,height=image.size
        lines,rings,complete,structure=[],[],False,None
    observed=inspect(path,page_number,None,MODEL/'eteks-door.npz',None)
    reader='visible PDF text'
    if not lines:
        from .raster_tags import read_raster_tags
        lines,reader=ocr_read(image,width,height)
        lines += read_raster_tags(image,observed.get('candidates',[]),width,height,lines)
    labels,marks=label_reads(lines)
    sx,sy=image.width/width,image.height/height
    shapes=assign_rooms(labels,rings,np.asarray(image.convert('L')),sx,sy,structure)
    rooms={k:{'label':v,**shapes.get(k,{})} for k,v in labels.items()}
    native=observed.get('evidence',{})
    frames={r['mark'].upper():r for r in observed.get('component_frames',[]) if r['bbox']}
    located_profiles={}
    if pdf and native and any(m not in frames for m in marks):
        import pymupdf
        with pymupdf.open(path) as doc:
            additional,located_profiles=marked_vector_profiles(doc[page_number-1],marks,native.get('swings',[]),frames)
            frames.update(additional)
    relations={}
    for mark,frame in frames.items():
        if len(marks.get(mark,[]))!=1:
            continue
        sides=polygon_sides(frame,rooms) or visible_sides(frame,rooms,structure,sx,sy)
        if sides:
            relations[mark]=sides
    return dict(source_sha256=sha256(path.read_bytes()).hexdigest(),page=page_number,width=width,height=height,
                coordinates='displayed_pdf_points' if pdf else 'image_pixels',reader=reader,
                native_text_complete=complete,lines=lines,rooms=rooms,marks=marks,frames=frames,relations=relations,
                located_profiles=located_profiles,
                suggestions=observed.get('candidates',[]),swing_evidence=native.get('swings',[]))
