"""Read a marked closed opening from opposing jambs and a connected leaf in a native vector PDF."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pymupdf


MARK=re.compile(r'^(?=.*\d)(?:DOOR|DR|D)(?:\d{1,5}[A-Z]?|(?:[-–._][A-Z0-9]{1,6}){1,4})$',re.I)


@dataclass
class Part:
    pts: np.ndarray
    center: np.ndarray
    short: float
    long: float
    pen: float
    color: np.ndarray


def bounds(points):
    return [*points.min(axis=0).tolist(),*points.max(axis=0).tolist()]


def components(page):
    parts=[]
    transform=page.rotation_matrix
    for path in page.get_drawings():
        color=path.get('color')
        fill=path.get('fill')
        if fill is None or color is None or path.get('fill_opacity',1)<.8 or np.mean(fill)<.8 or np.mean(fill)-np.mean(color)<.30:
            continue
        items=path['items']
        if len(items)==1 and items[0][0] in ('re','qu'):
            r=items[0][1]
            corners=(r.tl,r.tr,r.br,r.bl) if items[0][0]=='re' else (r.ul,r.ur,r.lr,r.ll)
        elif len(items) in (3,4) and all(item[0]=='l' for item in items):
            corners=[items[0][1],*(item[2] for item in items[:3])]
            if not all(abs(items[i-1][2]-items[i][1])<.08 for i in range(1,len(items))):
                continue
        else:
            continue
        xy=np.asarray([list(p*transform) for p in corners],dtype=float)
        u,v=xy[1]-xy[0],xy[2]-xy[1]
        a,b=np.linalg.norm(u),np.linalg.norm(v)
        if min(a,b)<.07 or abs(np.dot(u,v))/(a*b)>.08 or np.linalg.norm(xy[3]-(xy[0]+v))>max(.08,.025*max(a,b)):
            continue
        parts.append(Part(xy,xy.mean(axis=0),min(a,b),max(a,b),float(path.get('width') or 0),np.array(color)))
    return parts


def same_paint(a,b):
    return abs(a.pen-b.pen)<=max(.04,.15*max(a.pen,b.pen)) and np.max(np.abs(a.color-b.color))<.10


def extent(part,base,d,n):
    points=part.pts-base
    u,v=points@d,points@n
    return float(min(u)),float(max(u)),float(min(v)),float(max(v))


def supported_frame(a,b,parts,anchor,em):
    delta=b.center-a.center
    distance=np.linalg.norm(delta)
    jamb=(a.short+b.short)/2
    if distance<3*jamb or abs(a.short-b.short)>.22*jamb or not same_paint(a,b):
        return
    d=delta/distance
    n=np.array([-d[1],d[0]])
    middle=(a.center+b.center)/2
    along=abs((anchor-middle)@d); across=abs((anchor-middle)@n)
    if along>max(1.1*em,.11*distance) or across>max(1.1*em,1.7*jamb):
        return
    # A square on its diagonal presents an inflated projection. The leaf and jamb
    # must share their physical direction in addition to having similar pen/color.
    if (np.ptp(a.pts@d)>1.13*a.long) or (np.ptp(b.pts@d)>1.13*b.long):
        return
    local=[]
    for part in parts:
        if not same_paint(a,part):
            continue
        lo,hi,low,high=extent(part,a.center,d,n)
        if lo<-.7*jamb or hi>distance+.7*jamb or low<-.69*jamb or high>.69*jamb:
            continue
        local.append((part,lo,hi,low,high))
    posts=[(lo,hi) for p,lo,hi,v,w in local if p.long<=1.24*jamb and p.short>=.76*jamb]
    joins=[]
    for part,lo,hi,low,high in local:
        if hi-lo<2.2*jamb or high-low>1.15*jamb:
            continue
        # Each leaf is supported by posts at its two ends. A slider may show only
        # one half-panel, and multiple leaves may share the central post.
        if min((abs(lo-end) for _,end in posts),default=1e9)>.28*jamb:
            continue
        if min((abs(hi-start) for start,_ in posts),default=1e9)>.28*jamb:
            continue
        joins.append((lo,hi))
    if not joins:
        return
    covered=0.
    end=-float('inf')
    for lo,hi in sorted(joins):
        covered+=max(0,hi-max(lo,end))
        end=max(end,hi)
    coverage=covered/max(distance-jamb,1e-5)
    if coverage<(.34 if len(posts)>=3 else .73):
        return
    outer=np.vstack([a.pts,b.pts])
    length=float(np.ptp(outer@d))
    quality=along/max(jamb,.07*distance)+across/max(em,jamb)+.16*(1-min(1,coverage))
    return quality,dict(bbox=bounds(outer),length_pt=length,bearing_deg=float(np.rad2deg(np.arctan2(*d[::-1]))%180),
                        jamb_centers=[a.center.tolist(),b.center.tolist()],jamb_width_pt=jamb,
                        connected_leaves=len(joins),connected_span_fraction=float(coverage))


def inspect_page(page):
    parts=components(page)
    marks=[]
    for word in page.get_text('words'):
        label=word[4].strip()
        if not MARK.fullmatch(label):
            continue
        rect=pymupdf.Rect(word[:4])*page.rotation_matrix
        anchor=np.array([(rect.x0+rect.x1)/2,(rect.y0+rect.y1)/2])
        em=max(.5,min(rect.width,rect.height))
        radius=min(.3*max(page.rect.width,page.rect.height),max(16,32*em))
        nearby=[p for p in parts if np.linalg.norm(p.center-anchor)<radius]
        posts=[p for p in nearby if p.long/p.short<=1.22 and p.short<=2*em]
        options=[]
        for i,a in enumerate(posts):
            for b in posts[:i]:
                found=supported_frame(a,b,nearby,anchor,em)
                if found is not None:
                    options.append(found)
        distinct=[]
        for cost,row in sorted(options,key=lambda x:x[0]):
            if not any(np.max(np.abs(np.array(row['bbox'])-s[1]['bbox']))<.10 for s in distinct):
                distinct.append((cost,row))
        result=dict(mark=label,anchor=anchor.tolist(),bbox=None,status='unresolved')
        if distinct:
            if len(distinct)==1 or distinct[1][0]-distinct[0][0]>.28:
                result.update(**distinct[0][1],status='supported_jambs_and_leaf')
            else:
                result['status']='ambiguous'
        marks.append(result)
    return marks


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pdf',type=Path)
    p.add_argument('--page',type=int,default=1)
    args=p.parse_args()
    with pymupdf.open(args.pdf) as doc:
        if not 1<=args.page<=len(doc): p.error('Page outside document')
        result=inspect_page(doc[args.page-1])
    print(json.dumps(dict(page=args.page,marks=result,pdf_sha256=hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
          coordinates='Top-left displayed PDF points, xyxy. Bearing clockwise from positive x, modulo 180 degrees.'),allow_nan=False))
