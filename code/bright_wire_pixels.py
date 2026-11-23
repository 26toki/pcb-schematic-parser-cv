"""Recover thin bright-colored wire pixels only beside observed contacts.

No closing/dilation bridges gaps. Body masks remain hard exclusions. The pixels
are observations, not permission to merge a crossing without a junction dot.
"""
import cv2
import numpy as np
from PIL import Image

def restore_runs(mask,runs):
    for y,x,u in runs:
        if 0<=y<mask.shape[0]:mask[y,max(0,x):min(mask.shape[1],u)]=True
    return mask

def encode_runs(mask):
    result=[]
    for y in np.flatnonzero(mask.any(axis=1)):
        row=np.diff(np.pad(mask[y].astype(np.int8),(1,1)))
        result.extend([int(y),int(x),int(u)] for x,u in zip(np.flatnonzero(row==1),np.flatnonzero(row==-1)))
    return result

def recover_bright_pixels(rgb,blocked,anchors):
    """anchors: dicts with xy and horizontal/vertical direction from contacts."""
    rgb=np.asarray(rgb,dtype=np.uint8)
    dark=np.asarray(Image.fromarray(rgb).convert('L'))<160
    lo=rgb.min(axis=2);hi=rgb.max(axis=2)
    # Include a colored component's dark center too, to keep raster edge bands
    # together; only components with a genuinely missing long stroke qualify.
    colored=(lo<160)&(hi-lo>=60)
    for box in blocked:
        x,y,u,v=map(int,box);colored[max(0,y):min(rgb.shape[0],v),max(0,x):min(rgb.shape[1],u)]=False
    extra=np.zeros(dark.shape,dtype=bool)
    if not anchors or not colored.any():return extra,dict(accepted_components=0,recovered_pixels=0,seed_lines=[])
    n,labels,stats,_=cv2.connectedComponentsWithStats(colored.astype(np.uint8),8)
    selected=set();seeds=[]
    for direction,kernel in [('horizontal',(24,1)),('vertical',(1,24))]:
        opened=cv2.morphologyEx(colored.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,kernel))
        _,_,line_stats,centers=cv2.connectedComponentsWithStats(opened,8)
        for s,c in zip(line_stats[1:],centers[1:]):
            x,y,w,h,area=map(int,s);horizontal=direction=='horizontal'
            length,thickness=(w,h) if horizontal else (h,w)
            if length<24 or thickness>6 or length<8*thickness:continue
            axis=float(c[1 if horizontal else 0]);start=x if horizontal else y;end=x+w-1 if horizontal else y+h-1
            cross=round(axis)
            if horizontal:
                support=dark[max(0,cross-1):cross+2,start:end+1].any(axis=0)
            else:support=dark[start:end+1,max(0,cross-1):cross+2].any(axis=1)
            if support.mean()>.15:continue  # Preserve already-visible dark wires.
            near=[a for a in anchors if a['direction']==direction and
                abs(a['xy'][1 if horizontal else 0]-axis)<=3 and
                max(start-a['xy'][0 if horizontal else 1],a['xy'][0 if horizontal else 1]-end,0)<=8]
            if not near:continue
            identities=set(int(i) for i in np.unique(labels[y:y+h,x:x+w][opened[y:y+h,x:x+w]>0]) if i)
            selected.update(identities)
            seeds.append(dict(direction=direction,axis=axis,start=start,end=end,observed_contact_support=len(near)))
    if selected:
        keep=np.zeros(n,dtype=bool);keep[list(selected)]=True;extra=keep[labels]&~dark
    return extra,dict(accepted_components=len(selected),recovered_pixels=int(extra.sum()),seed_lines=seeds,
        policy='Thin colored stroke >=24px, existing inline contact, missing from dark mask; no gap filling')
