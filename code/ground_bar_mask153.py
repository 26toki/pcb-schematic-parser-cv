"""Exclude supported ground-symbol bars from wire tracing; preserve wire colors."""
import re,itertools
import numpy as np
import cv2
def ground_bar_mask(image,items,excluded_boxes=()):
    a=np.asarray(image.convert('RGB')).astype(np.int16);red=((a[:,:,0]-a[:,:,1]>35)&(a[:,:,0]-a[:,:,2]>35)&(a[:,:,0]>70)).astype('uint8');mask=np.zeros(red.shape,bool);evidence=[]
    for t in items:
        # Restrict this repair to crowded symbols whose text OCR merged.
        # Isolated grounds remain on the validated existing tracing path.
        if t.get('confidence',0)<.95 or not re.fullmatch(r'(?:[ADP]?GND){2,4}',t['text']):continue
        tb=t['bbox_xyxy'];tcx,tcy=(tb[0]+tb[2])/2,(tb[1]+tb[3])/2
        if any(x<=tcx<=u and y<=tcy<=v for x,y,u,v in excluded_boxes):continue
        b=t['bbox_xyxy'];x=max(0,int(b[0])-75);y=max(0,int(b[1])-75);u=min(red.shape[1],int(b[2])+76);v=min(red.shape[0],int(b[3])+76)
        for axis in (0,1):
            opened=cv2.morphologyEx(red[y:v,x:u],cv2.MORPH_OPEN,np.ones((1,5) if axis==0 else (5,1),np.uint8));n,labels,stats,centers=cv2.connectedComponentsWithStats(opened,8);bars=[]
            for i in range(1,n):
                px,py,w,h,area=map(int,stats[i]);length,thickness=(w,h) if axis==0 else (h,w)
                if 4<=length<=100 and thickness<=3:bars.append(dict(id=i,length=length,axis=float(centers[i][1-axis]),bbox=[px+x,py+y,px+w+x,py+h+y]))
            for triple in itertools.combinations(bars,3):
                q=sorted(triple,key=lambda s:s['axis']);ds=[q[1]['axis']-q[0]['axis'],q[2]['axis']-q[1]['axis']]
                if not all(3<=d<=14 for d in ds) or max(ds)/min(ds)>1.5:continue
                lengths=[z['length'] for z in q]
                if not (lengths[0]>lengths[1]>lengths[2] or lengths[0]<lengths[1]<lengths[2]):continue
                if min(lengths)>.8*max(lengths):continue
                # Each bar must intersect the same perpendicular center axis.
                lo=max(z['bbox'][axis] for z in q);hi=min(z['bbox'][axis+2] for z in q)
                if lo>=hi:continue
                for z in q:mask[y:v,x:u]|=labels==z['id']
                evidence.append(dict(ocr_id=t['id'],bars=[z['bbox'] for z in q]))
    # A red glyph can overpaint a few pixels of an otherwise visible green
    # wire. Preserve that crossing only with green runs on BOTH sides; this
    # does not reconnect the removed perpendicular symbol stroke.
    green=(a[:,:,1]-a[:,:,0]>30)&(a[:,:,1]-a[:,:,2]>30)&(a[:,:,1]>60)
    restore=np.zeros_like(mask)
    for yy,xx in zip(*np.where(mask)):
        for dx,dy in ((1,0),(0,1)):
            ok=True
            for sign in (-1,1):
                found=False
                for gap in range(1,5):
                    x,y=xx+dx*sign*gap,yy+dy*sign*gap
                    if not (0<=x<mask.shape[1] and 0<=y<mask.shape[0]):break
                    if green[y,x]:
                        tail=[(x+dx*sign*k,y+dy*sign*k) for k in range(4)]
                        found=all(0<=px<mask.shape[1] and 0<=py<mask.shape[0] and green[py,px] for px,py in tail);break
                    if not mask[y,x]:break
                ok&=found
            if ok:restore[yy,xx]=True
    mask[restore]=False
    evidence.append(dict(preserved_green_crossing_pixels=int(restore.sum())))
    return mask,evidence
