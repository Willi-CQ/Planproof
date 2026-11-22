"""PlanProof original-symbol detector. Boxes are [x0,y0,x1,y1] in caller image pixels."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from detection_boxes import image_array, iou, merge, positions

NAMES = ("airconditioner", "table", "tv_cabinet", "single_door")
STRIDE = 4


def conv(src, dst, stride=1):
    return nn.Sequential(nn.Conv2d(src,dst,3,stride,1,bias=False),nn.BatchNorm2d(dst),nn.ReLU())


class Detector(nn.Module):
    """Stride-four center heatmaps with two log-sides and a center offset."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv(1,12,2),conv(12,20,2),conv(20,20))
        self.mid = nn.Sequential(conv(20,32,2),conv(32,32))
        self.deep = nn.Sequential(conv(32,48,2),conv(48,48),conv(48,40))
        self.fuse = nn.Sequential(conv(20+32+40,32),conv(32,24))
        self.heat = nn.Conv2d(24,4,1)
        self.geom = nn.Conv2d(24,4,1)
        nn.init.constant_(self.heat.bias,-2.18)

    def forward(self, x):
        a=self.stem(x)
        b=self.mid(a)
        c=self.deep(b)
        merged=torch.cat((a,F.interpolate(b,size=a.shape[-2:],mode="nearest"),
                          F.interpolate(c,size=a.shape[-2:],mode="nearest")),dim=1)
        h=self.fuse(merged)
        return self.heat(h),self.geom(h)


def save_model(model, path, threshold):
    fields={name:parameter.detach().cpu().numpy() for name,parameter in model.state_dict().items()}
    np.savez_compressed(path,**fields,decision_threshold=np.float32(threshold),version=np.int32(1))


def load_model(path):
    net=Detector()
    with np.load(path,allow_pickle=False) as a:
        if a["version"].item()!=1:
            raise ValueError("Unsupported weight format")
        net.load_state_dict({k:torch.from_numpy(a[k].copy()) for k in net.state_dict()})
        threshold=float(a["decision_threshold"])
    return net.eval(),threshold


@torch.inference_mode()
def view_predictions(model, image, threshold):
    arr=image_array(image)
    h,w=arr.shape
    pad=(0,(-w)%16,0,(-h)%16)
    x=F.pad((255-torch.from_numpy(arr)).float()[None,None]/255,pad)
    logits,g=model(x)
    heat=logits.sigmoid()[0]
    best=F.max_pool2d(heat.unsqueeze(0),3,1,1)[0]
    cats,ys,xs=torch.where((heat>=threshold)&(heat==best))
    values=heat[cats,ys,xs].tolist()
    geometry=g[0,:,ys,xs].T.numpy()
    output=[]
    for cat,y,x,conf,(logw,logh,fx,fy) in zip(cats.tolist(),ys.tolist(),xs.tolist(),values,geometry):
        cx,cy=(x+float(np.clip(fx,0,1)))*4,(y+float(np.clip(fy,0,1)))*4
        if cx>=w or cy>=h:
            continue
        bw,bh=np.exp(np.clip([logw,logh],0,5))*4
        bounds=[max(0,cx-float(bw)/2),max(0,cy-float(bh)/2),min(w,cx+float(bw)/2),min(h,cy+float(bh)/2)]
        if bounds[2]>bounds[0] and bounds[3]>bounds[1]:
            output.append({"category":NAMES[cat],"bbox":bounds,"confidence":conf})
    return output


def predict(model, image, threshold, mode="global"):
    """Global scales long side to at most 448; tiled adds 448px windows with 20% overlap."""
    if mode not in {"global","tiled"}:
        raise ValueError("unknown inference mode")
    w,h=image.size
    ratio=min(1.,448/max(w,h))
    resized=image.resize((max(1,round(w*ratio)),max(1,round(h*ratio))),Image.Resampling.BILINEAR) if ratio<1 else image
    result=view_predictions(model,resized,threshold)
    for row in result:
        row["bbox"]=[v*extent/new for v,extent,new in zip(row["bbox"],[w,h,w,h],
                     [resized.width,resized.height]*2)]
    if mode=="tiled" and (w>448 or h>448):
        for y in positions(h,448,.2):
            for x in positions(w,448,.2):
                tile=image.crop((x,y,min(w,x+448),min(h,y+448)))
                for row in view_predictions(model,tile,threshold):
                    row["bbox"]=[v+o for v,o in zip(row["bbox"],[x,y,x,y])]
                    result.append(row)
    return merge(result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights")
    p.add_argument("image")
    p.add_argument("--mode",choices=("global","tiled"),default="global")
    args=p.parse_args()
    net,cutoff=load_model(args.weights)
    with Image.open(args.image) as im:
        print(json.dumps(predict(net,im,cutoff,args.mode),indent=2))

