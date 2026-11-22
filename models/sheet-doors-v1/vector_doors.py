"""Trace visible swing-door geometry in a vector PDF. Displayed coordinates are PDF points."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pymupdf
from scipy.spatial import cKDTree


def xy(z):
    return [round(z.real,4),round(z.imag,4)]


def box(points):
    return [round(min(p.real for p in points),4),round(min(p.imag for p in points),4),
            round(max(p.real for p in points),4),round(max(p.imag for p in points),4)]


def inside(z,scope):
    return scope[0]<=z.real<=scope[2] and scope[1]<=z.imag<=scope[3]


def ang(u,v):
    return math.degrees(math.atan2((u.conjugate()*v).imag,(u.conjugate()*v).real))


def primitives(page):
    matrix=page.rotation_matrix
    def point(p):
        q=p*matrix
        return complex(q.x,q.y)
    strokes=[]
    paths=[]
    profiles=[]
    for path in page.get_drawings():
        chain=[]
        for item in path['items']:
            op=item[0]
            if op=='l':
                pts=[point(p) for p in item[1:3]]
            elif op=='c':
                a,b,c,d=map(point,item[1:5])
                pts=[(1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d for t in np.linspace(0,1,9)]
            elif op in ('re','qu'):
                r=item[1]
                corners=[point(p) for p in (r.tl,r.tr,r.br,r.bl)] if op=='re' else [point(p) for p in (r.ul,r.ur,r.lr,r.ll)]
                profiles.append(corners)
                strokes.extend(zip(corners,corners[1:]+corners[:1]))
                continue
            else:
                continue
            if abs(pts[0]-pts[-1])<.002:
                continue
            if op=='l':
                strokes.append((pts[0],pts[-1]))
            if chain and abs(chain[-1]-pts[0])>.09:
                paths.append(chain)
                chain=[]
            chain.extend(pts[1:] if chain else pts)
        if chain:
            paths.append(chain)
            if len(chain) in (4,5) and (path.get('closePath') or abs(chain[0]-chain[-1])<.1):
                profiles.append(chain[:4])
    return strokes,paths,profiles


def curve_runs(points):
    if len(points)<4:
        return
    begin=0
    sign=0
    for i in range(1,len(points)-1):
        turn=ang(points[i]-points[i-1],points[i+1]-points[i])
        now=0 if abs(turn)<.8 else (1 if turn>0 else -1)
        if abs(turn)>43 or (now and sign and now!=sign):
            if i+1-begin>=4:
                yield points[begin:i+1]
            begin=i
            sign=0
        elif now:
            sign=now
    if len(points)-begin>=4:
        yield points[begin:]


def fit_arc(pts,page_min):
    pts=np.array(pts,dtype=complex)
    mean=pts.mean()
    v=pts-mean
    aa=np.column_stack([2*v.real,2*v.imag,np.ones(len(v))])
    coef,_,rank,_=np.linalg.lstsq(aa,np.abs(v)**2,rcond=None)
    if rank<3:
        return
    hinge=complex(coef[0],coef[1])+mean
    rr=np.abs(pts-hinge)
    r=float(np.mean(rr))
    if not 2.5<=r<=.25*page_min or np.max(np.abs(rr-r))>.045*r:
        return
    angles=np.unwrap(np.angle(pts-hinge))
    sweep=abs(math.degrees(angles[-1]-angles[0]))
    if 65<=sweep<=114:
        return hinge,r,sweep


def joined_strokes(lines):
    """Join quarter-arcs exploded into separate polyline segments by PDF exporters."""
    if not lines:
        return
    bins=defaultdict(list)
    def cell(p):
        return round(p.real/.08),round(p.imag/.08)
    for i,(a,b) in enumerate(lines):
        bins[cell(a)].append((i,0))
        bins[cell(b)].append((i,1))
    connections={}
    for adjacent in bins.values():
        if len(adjacent)!=2:
            continue
        (i,k),(j,l)=adjacent
        if i==j:
            continue
        u=lines[i][k]-lines[i][1-k]
        v=lines[j][1-l]-lines[j][l]
        if abs(ang(u,v))<41 and .38<abs(u)/max(.001,abs(v))<2.65:
            connections[i,k]=(j,l)
            connections[j,l]=(i,k)
    seen=set()
    for i,line in enumerate(lines):
        if i in seen or ((i,0) in connections and (i,1) in connections):
            continue
        k=0 if (i,0) not in connections else 1
        pts=[line[k]]
        while i not in seen:
            seen.add(i)
            pts.append(lines[i][1-k])
            following=connections.get((i,1-k))
            if not following:
                break
            i,k=following
        if len(pts)>=4:
            yield pts


def swing_candidates(lines,paths,page):
    if not lines:
        return []
    endpoints=np.array([xy(p) for line in lines for p in line])
    tree=cKDTree(endpoints)
    found=[]
    for pts in list(paths)+list(joined_strokes(lines)):
        for run in curve_runs(pts):
            arc=fit_arc(run,min(page.width,page.height))
            if arc is None:
                continue
            center,r,sweep=arc
            if not inside(center,[0,0,page.width,page.height]):
                continue
            nearby=tree.query_ball_point(xy(center),max(.75,.19*r))
            options=[]
            for idx in nearby:
                a,b=lines[idx//2][::1 if idx%2==0 else -1]
                for end,closed in ((run[0],run[-1]),(run[-1],run[0])):
                    error=max(abs(a-center),abs(b-end))/r
                    if error<=.18 and .76<=abs(b-a)/r<=1.25:
                        options.append((error,a,b,end,closed))
            if not options:
                continue
            error,a,b,end,closed=min(options,key=lambda row:row[0])
            # The wall continuation is optional evidence. The leaf and circular swing are mandatory.
            wall=[]
            for pivot in (center,closed):
                for ii in tree.query_ball_point(xy(pivot),max(.7,.14*r)):
                    p,q=lines[ii//2][::1 if ii%2==0 else -1]
                    if abs(q-p)>.30*r and abs(ang(q-p,closed-center))%180<16:
                        if min(abs(q-end),abs(q-center))>.22*r:
                            wall=[xy(p),xy(q)]
                            break
                if wall:
                    break
            found.append(dict(hinge=center,open=end,closed=closed,r=r,sweep=sweep,wall=wall,
                              bbox=box([center,a,b,*run]),quality=round(1-error,5)))
    unique=[]
    for row in sorted(found,key=lambda row:-row['quality']):
        if not any(abs(row['hinge']-s['hinge'])<.17*row['r'] and abs(row['open']-s['open'])<.19*row['r']
                   and abs(row['closed']-s['closed'])<.19*row['r'] for s in unique):
            unique.append(row)
    paired=set()
    for i,row in enumerate(unique):
        for j in range(i):
            other=unique[j]
            if (abs(row['closed']-other['closed'])<.25*max(row['r'],other['r']) and
                abs(ang(row['closed']-row['hinge'],other['closed']-other['hinge']))>150 and
                abs(ang(row['open']-row['hinge'],other['open']-other['hinge']))<35):
                paired.update((i,j))
    return [dict(category='single_swing_candidate',bbox=r['bbox'],hinge=xy(r['hinge']),leaf_tip=xy(r['open']),
                 jamb=xy(r['closed']),radius_pt=round(r['r'],4),sweep_deg=round(r['sweep'],3),
                 nearby_wall=r['wall'],fit_score=r['quality'],paired_leaf=i in paired)
            for i,r in enumerate(unique)]


MARK=re.compile(r'^(?=.*\d)(?:DOOR|DR|D)(?:\d{1,5}[A-Z]?|(?:[-–._][A-Z0-9]{1,6}){1,4})$',re.I)


def marked_openings(page,lines,quads,schedule=None):
    """Bind text IDs to nearby thin rectangular vector profiles, leaving ambiguous marks unresolved."""
    marks=[]
    for word in page.get_text('words'):
        label=word[4].strip()
        if MARK.fullmatch(label):
            rect=pymupdf.Rect(word[:4])*page.rotation_matrix
            marks.append((label,rect,complex((rect.x0+rect.x1)/2,(rect.y0+rect.y1)/2)))
    candidates=[]
    # CAD exporters also stroke the opposite sides of a rectangular leaf as independent lines.
    segments=[(a,b) for a,b in lines if 3<=abs(b-a)<=.25*min(page.rect.width,page.rect.height)]
    if segments and marks:
        midpoints=[(a+b)/2 for a,b in segments]
        tree=cKDTree([xy(m) for m in midpoints])
        used=set()
        for _,rect,pos in marks:
            em=max(.5,min(rect.width,rect.height))
            for i in tree.query_ball_point(xy(pos),20*em):
                if i in used:
                    continue
                used.add(i)
                a,b=segments[i]
                long=abs(a-b)
                for j in tree.query_ball_point(xy(midpoints[i]),.31*long):
                    if j==i:
                        continue
                    p,q=segments[j]
                    if abs(p-b)<abs(p-a):
                        p,q=q,p
                    d=(p-a)/((b-a)/long)
                    e=(q-b)/((b-a)/long)
                    if (abs(d.real)<.07*long and abs(e.real)<.07*long and
                        .024*long<abs(d.imag)<.30*long and abs(d.imag-e.imag)<.035*long):
                        quads.append([a,b,q,p])
    for q in quads:
        u,v=q[1]-q[0],q[2]-q[1]
        if min(abs(u),abs(v))<.15 or abs(abs(ang(u,v))-90)>7:
            continue
        if max(abs((q[3]-q[2])+u),abs((q[0]-q[3])+v))>max(.2,.06*max(abs(u),abs(v))):
            continue
        long,short=sorted((abs(u),abs(v)),reverse=True)
        if not (3.2<=long/short<=44 and 3<=long<=.25*min(page.rect.width,page.rect.height)):
            continue
        candidates.append((sum(q)/4,long,short,box(q)))
    out=[]
    for label,rect,anchor in marks:
        matches=[]
        for mid,long,short,b in candidates:
            # A door identifier has a short typographic dimension. Width-dependent search tolerates rotated text.
            em=max(.5,min(rect.width,rect.height))
            if 1.15*em<=long<=19*em:
                distance=abs(mid-anchor)
                if distance<=max(3.4*em,1.25*long):
                    matches.append((distance/max(em,long*.6),b))
        matches.sort(key=lambda x:x[0])
        distinct=[]
        for score,b in matches:
            if not any(max(abs(x-y) for x,y in zip(b,other))<.5 for _,other in distinct):
                distinct.append((score,b))
        selected=bool(distinct and (len(distinct)==1 or distinct[1][0]-distinct[0][0]>.15))
        kind=(schedule or {}).get(label)
        out.append(dict(mark=label,text_bbox=list(rect),anchor=xy(anchor),bbox=distinct[0][1] if selected else None,
                        status='profile' if selected else 'ambiguous' if distinct else 'unresolved',schedule_class=kind))
    return out


def inspect_page(page,region=None,schedule=None):
    scope=list(region or page.rect)
    if len(scope)!=4 or not (0<=scope[0]<scope[2]<=page.rect.width and 0<=scope[1]<scope[3]<=page.rect.height):
        raise ValueError('Region must lie in displayed top-left page points')
    lines,paths,quads=primitives(page)
    swings=swing_candidates(lines,paths,page.rect)
    swings=[r for r in swings if inside(complex((r['bbox'][0]+r['bbox'][2])/2,(r['bbox'][1]+r['bbox'][3])/2),scope)]
    marks=[r for r in marked_openings(page,lines,quads,schedule) if inside(complex(*r['anchor']),scope)]
    return dict(page=page.number+1,width=page.rect.width,height=page.rect.height,rotation=page.rotation,region=scope,
                coordinates='displayed PDF points, top-left origin, xyxy',vector_strokes=len(lines),
                swings=[s for s in swings if not s['paired_leaf']],paired_leaves=[s for s in swings if s['paired_leaf']],
                swing_status='candidates' if any(not s['paired_leaf'] for s in swings) else 'no_supported_evidence',
                marked_openings=marks,mark_status='marks_found' if marks else 'no_supported_evidence')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pdf',type=Path)
    p.add_argument('--page',type=int,default=1)
    p.add_argument('--region',nargs=4,type=float)
    p.add_argument('--schedule',type=Path,help='Optional user-provided JSON mapping printed door marks to class descriptions')
    args=p.parse_args()
    schedule=json.loads(args.schedule.read_text()) if args.schedule else None
    with pymupdf.open(args.pdf) as doc:
        if not 1<=args.page<=len(doc):
            p.error('Page number is outside this PDF')
        result=inspect_page(doc[args.page-1],args.region,schedule)
    result['pdf_sha256']=hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    main()
