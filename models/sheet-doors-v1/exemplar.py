"""Find other instances of a symbol given one example on the input sheet."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return overlap/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)


def edge_layers(gray):
    smooth=cv2.GaussianBlur(gray,(3,3),.55)
    edges=cv2.Canny(smooth,38,100)
    dx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
    bins=np.round(cv2.phase(dx,dy,angleInDegrees=True)/45).astype('uint8')%4
    return [(edges>0)&(bins==i) for i in range(4)]


def compatible(image):
    layers=edge_layers(image)
    result=[]
    for layer in layers:
        if layer.any():
            dist=cv2.distanceTransform((~layer).astype('uint8'),cv2.DIST_L2,3)
            result.append(np.exp(-.45*dist*dist))
        else:
            result.append(np.zeros(image.shape,np.float32))
    return result


def find_other(image,seed,cut=.6):
    cv2.setNumThreads(2)
    white=Image.new('RGBA',image.size,'white')
    white.alpha_composite(image.convert('RGBA'))
    gray=np.asarray(white.convert('L'))
    h,w=gray.shape
    l,t,r,b=map(float,seed)
    if not (0<=l<r<=w and 0<=t<b<=h and min(r-l,b-t)>=5):
        raise ValueError('Seed must be a valid original-page box of at least five pixels per side')
    ratio=min(3.,36/max(r-l,b-t),3000/max(w,h))
    sw,sh=max(1,round(w*ratio)),max(1,round(h*ratio))
    sx,sy=sw/w,sh/h
    page=cv2.resize(gray,(sw,sh),interpolation=cv2.INTER_AREA if ratio<1 else cv2.INTER_LINEAR)
    x0,y0=max(0,round(l*sx)),max(0,round(t*sy))
    x1,y1=min(sw,round(r*sx)),min(sh,round(b*sy))
    patch=page[y0:y1,x0:x1]
    if min(patch.shape)<5:
        return []
    page_layers=compatible(page)
    # Compensate for clutter and ruling that happens to pass through a template's edges.
    ink=(page<195).astype(np.float32)
    summed=cv2.integral(ink)
    raw=[]
    for turn in range(4):
        rotated=np.rot90(patch,turn)
        for mirror in (False,True):
            view=np.fliplr(rotated) if mirror else rotated
            for factor in (.78,.9,1.,1.12,1.28):
                a,bw=view.shape
                hh,ww=max(5,round(a*factor)),max(5,round(bw*factor))
                if hh>sh or ww>sw:
                    continue
                tp=cv2.resize(view,(ww,hh),interpolation=cv2.INTER_LINEAR)
                layers=edge_layers(tp)
                count=sum(int(m.sum()) for m in layers)
                if count<12: continue
                score=np.zeros((sh-hh+1,sw-ww+1),np.float32)
                for m,match in zip(layers,page_layers):
                    if m.any():
                        score+=cv2.matchTemplate(match,m.astype(np.float32),cv2.TM_CCORR)
                score/=count
                expected=float((tp<195).mean())
                density=(summed[hh:,ww:]-summed[:-hh,ww:]-summed[hh:,:-ww]+summed[:-hh,:-ww])/(hh*ww)
                score-=.58*np.abs(density-expected).astype(np.float32)
                kernel=max(5,round(min(hh,ww)*.52))|1
                peaks=(score>=cut)&(score>=cv2.dilate(score,np.ones((kernel,kernel),np.uint8))-1e-7)
                ys,xs=np.where(peaks)
                for y,x in zip(ys,xs):
                    box=[x/sx,y/sy,(x+ww)/sx,(y+hh)/sy]
                    if iou(box,seed)<.24:
                        raw.append(dict(category='single_door',bbox=box,confidence=float(score[y,x])))
    keep=[]
    for row in sorted(raw,key=lambda v:-v['confidence']):
        if all(iou(row['bbox'],k['bbox'])<.29 for k in keep):
            keep.append(row)
    return keep


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',type=Path)
    p.add_argument('--example',nargs=4,type=float,required=True,metavar=('LEFT','TOP','RIGHT','BOTTOM'))
    p.add_argument('--configuration',type=Path,required=True)
    a=p.parse_args()
    config=json.loads(a.configuration.read_text())
    with Image.open(a.image) as im:
        rows=find_other(im,a.example,config['threshold'])
    print(json.dumps(dict(example=a.example,additional=rows,proposed_total=1+len(rows))))
