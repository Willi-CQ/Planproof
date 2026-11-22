"""Resolve enclosure ownership and sample the two physical sides of an opening."""
from collections import Counter
import math

import cv2
import numpy as np
from shapely.geometry import Point, Polygon, box as rect


def center(box):
    return ((box[0]+box[2])/2,(box[1]+box[3])/2)


def bbox(points):
    return [min(p[0] for p in points),min(p[1] for p in points),
            max(p[0] for p in points),max(p[1] for p in points)]


def white_enclosures(gray,labels,scale_x,scale_y):
    """Keep enclosed regions that are separated by actual strokes on the page."""
    _, ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # A single global exterior contour would hide every room inside the border.
    _,groups,stats,_=cv2.connectedComponentsWithStats(255-ink,connectivity=4)
    h,w=gray.shape
    selected={}
    for key,label in labels.items():
        box=label['bbox']
        x,y=center(box)
        x,y=round(x*scale_x),round(y*scale_y)
        radius=max(4,int((box[3]-box[1])*scale_y*.9))
        region=groups[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]
        ids=Counter(int(n) for n in region.ravel() if n>0)
        for index,_ in ids.most_common():
            left,top,ww,hh,area=stats[index]
            if min(left,top,w-left-ww,h-top-hh)<=0 or area<max(30,5*radius**2) or area>.90*w*h:
                continue
            cropped=np.uint8(groups[top:top+hh,left:left+ww]==index)
            contours,_=cv2.findContours(cropped,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour=max(contours,key=cv2.contourArea)
            points=cv2.approxPolyDP(contour,1.2,True).reshape(-1,2)
            if len(points)<3:
                continue
            poly=Polygon([((xx+left)/scale_x,(yy+top)/scale_y) for xx,yy in points])
            if (poly.is_valid and poly.area>0 and
                    not any(poly.covers(Point(center(other['bbox']))) for n,other in labels.items() if n!=key)):
                selected[key]=(poly,index)
                break
    counts=Counter(index for _,index in selected.values())
    return {key:poly for key,(poly,i) in selected.items() if counts[i]==1}


def structural_mask(page,width,height):
    """Project closed joinery strokes and heavy wall pens, excluding text and dimension hairlines."""
    import pymupdf
    paths=page.get_drawings()
    sx,sy=width/page.rect.width,height/page.rect.height
    strokes=[float(p.get('width') or 0) for p in paths if p.get('color') and max(p['color'])<.5]
    mask=np.full((height,width),255,np.uint8)
    if not strokes:return None
    thick=max(.7,float(np.percentile(strokes,85)))
    matrix=page.rotation_matrix
    def coords(p):
        q=p*matrix
        return (round(q.x*sx),round(q.y*sy))
    for p in paths:
        if not p.get('color') or max(p['color'])>.55 or (p.get('stroke_opacity') or 0)<.5:
            continue
        pen=float(p.get('width') or 0)
        if pen<thick-.01 and (not p.get('fill') or pen<.45):
            continue
        size=max(1,int(round(pen*max(sx,sy))))
        for item in p['items']:
            if item[0]=='l':cv2.line(mask,coords(item[1]),coords(item[2]),0,size)
            if item[0] in ('re','qu'):
                shape=item[1]
                pts=np.array([coords(pt) for pt in (shape.tl,shape.tr,shape.br,shape.bl)],np.int32) if item[0]=='re' else np.array([coords(pt) for pt in (shape.ul,shape.ur,shape.lr,shape.ll)],np.int32)
                cv2.polylines(mask,[pts],True,0,size)
    return mask


def visible_sides(frame,rooms,mask,sx,sy):
    """Use two short probes outside an opening and unobstructed sightlines to named interiors."""
    if mask is None:return None
    cx,cy=center(frame['bbox']);length=frame['length_pt']
    theta=math.radians(frame['bearing_deg']);nx,ny=-math.sin(theta),math.cos(theta)
    height,width=mask.shape
    sides=[]
    for sign in (-1,1):
        point=(cx+sign*nx*.21*length,cy+sign*ny*.21*length)
        possible=[]
        for name,room in rooms.items():
            x,y=center(room['label']['bbox'])
            if sign*((x-cx)*nx+(y-cy)*ny)<=0:
                continue
            d=math.dist(point,(x,y))
            if d>12*length:continue
            clear=0
            for delta in (-.09,0,.09):
                start=(point[0]+length*delta*math.cos(theta),point[1]+length*delta*math.sin(theta))
                n=max(3,int(d*max(sx,sy)*1.3))
                xx=np.clip(np.rint(np.linspace(start[0]*sx,x*sx,n)).astype(int),0,width-1)
                yy=np.clip(np.rint(np.linspace(start[1]*sy,y*sy,n)).astype(int),0,height-1)
                if not np.any(mask[yy,xx]==0):clear+=1
            if clear>=2:possible.append((d,name))
        possible.sort()
        if not possible or (len(possible)>1 and possible[1][0]<1.25*possible[0][0]):
            return None
        sides.append((possible[0][1],list(point)))
    if sides[0][0]==sides[1][0]:return None
    return {'rooms':sorted(name for name,_ in sides),'basis':'unobstructed opposite-side probes to room labels',
            'probes':{name:point for name,point in sides}}


def assign_rooms(labels,original,gray,scale_x,scale_y,structure=None):
    from shapely.strtree import STRtree
    centers={key:Point(center(value['bbox'])) for key,value in labels.items()}
    shapes=[]
    explicit=[]
    for ring in original:
        poly=Polygon(ring['points'] if isinstance(ring,dict) else ring)
        if poly.is_valid and poly.area>0:
            shapes.append(poly)
            explicit.append(isinstance(ring,dict) and ring['explicit'])
    tree=STRtree(shapes)
    white=white_enclosures(gray,labels,scale_x,scale_y) if labels else {}
    physical=white_enclosures(structure,labels,scale_x,scale_y) if labels and structure is not None else {}
    result={}
    for key,anchor in labels.items():
        point=centers[key]
        text_area=(anchor['bbox'][2]-anchor['bbox'][0])*(anchor['bbox'][3]-anchor['bbox'][1])
        choices=[]
        for idx in tree.query(point,predicate='intersects'):
            p=shapes[int(idx)]
            if (p.area<5*text_area or not p.covers(rect(*anchor['bbox'])) or
                    any(p.covers(pt) for name,pt in centers.items() if name!=key)):
                continue
            choices.append((not explicit[int(idx)],p.area,p))
        # Overlaying a swing or a dimension may subdivide a room into smaller visible
        # faces. Prefer its directly authored closed outline if it owns only this label.
        shape=min(choices,key=lambda entry:entry[:2])[2] if choices else physical.get(key,white.get(key))
        if shape is not None:
            result[key]={'polygon':[list(x) for x in shape.exterior.coords],'bbox':list(shape.bounds),
                         'basis':'closed vector boundary' if choices else 'enclosed structural strokes' if key in physical else 'enclosed raster component'}
    return result


def room_contacts(frame,rooms):
    """Opposite probes must land in unique room interiors on both sides."""
    box=frame['bbox']
    cx,cy=center(box)
    bearing=frame.get('bearing_deg')
    if bearing is None:
        return []
    theta=math.radians(bearing)
    tangent=(math.cos(theta),math.sin(theta))
    norm=(-tangent[1],tangent[0])
    length=float(frame['length'])
    shapes={key:Polygon(value['polygon']) for key,value in rooms.items() if 'polygon' in value}
    pairs=Counter()
    for fraction in (.08,.15,.24,.34):
        for lateral in (-.15,0,.15):
            sides=[]
            for sign in (-1,1):
                pt=Point(cx+length*(lateral*tangent[0]+sign*fraction*norm[0]),
                         cy+length*(lateral*tangent[1]+sign*fraction*norm[1]))
                inside=[key for key,p in shapes.items() if p.contains(pt)]
                sides.append(inside[0] if len(inside)==1 else None)
            if None not in sides and sides[0]!=sides[1]:
                pairs[tuple(sorted(sides))]+=1
    candidates=[pair for pair,n in pairs.items() if n>=2]
    return list(candidates[0]) if len(candidates)==1 else []


def tagged_boundary_contact(mark,rooms):
    """Find directly opposed interiors on a shared straight boundary at the printed tag."""
    from shapely.geometry import LineString
    cx,cy=center(mark['bbox'])
    point=Point(cx,cy)
    em=max(1,min(mark['bbox'][2]-mark['bbox'][0],mark['bbox'][3]-mark['bbox'][1]))
    choices=[]
    for name,room in rooms.items():
        ring=room.get('polygon',[])
        edges=[LineString([a,b]) for a,b in zip(ring,ring[1:]) if math.dist(a,b)>=2*em]
        if edges:
            edge=min(edges,key=lambda e:e.distance(point))
            if edge.distance(point)<=1.25*em:
                choices.append((name,edge,Polygon(ring)))
    found=set()
    for i,(a,edge,pa) in enumerate(choices):
        ends=list(edge.coords)
        theta=math.atan2(ends[1][1]-ends[0][1],ends[1][0]-ends[0][0])
        for b,second,pb in choices[:i]:
            ends2=list(second.coords)
            alpha=math.atan2(ends2[1][1]-ends2[0][1],ends2[1][0]-ends2[0][0])
            if abs(math.sin(theta-alpha))>.16:
                continue
            positive=Point(cx-1.6*em*math.sin(theta),cy+1.6*em*math.cos(theta))
            negative=Point(cx+1.6*em*math.sin(theta),cy-1.6*em*math.cos(theta))
            if (pa.contains(positive) and pb.contains(negative)) or (pb.contains(positive) and pa.contains(negative)):
                found.add(tuple(sorted((a,b))))
    return list(next(iter(found))) if len(found)==1 else []
