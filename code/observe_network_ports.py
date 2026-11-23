"""Observe single-contact pentagonal network ports from pixels and OCR only."""
from collections import Counter
import math,re
import cv2
import numpy as np


def observe_ports(rgb,items,excluded_boxes,local_reader=None):
    a=np.asarray(rgb).astype(np.int16)
    red=(a[:,:,0]-a[:,:,1]>45)&(a[:,:,0]-a[:,:,2]>45)&(a[:,:,0]>75)
    green=(a[:,:,1]-a[:,:,0]>35)&(a[:,:,1]-a[:,:,2]>35)&(a[:,:,1]>60)
    wire=red|green
    # Close only sub-stroke raster gaps; keep wire validation on original pixels.
    closed=cv2.morphologyEx(red.astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    inside=lambda p,b:b[0]<=p[0]<=b[2] and b[1]<=p[1]<=b[3]
    def branch(point,axis,sign):
        samples=[]
        for step in range(3,13):
            x,y=map(round,point);x+=sign*step if axis==0 else 0;y+=sign*step if axis==1 else 0
            if not 1<=x<wire.shape[1]-1 or not 1<=y<wire.shape[0]-1:return False
            samples.append(wire[y-1:y+2,x].any() if axis==0 else wire[y,x-1:x+2].any())
        return sum(samples)>=8
    proposals=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if not (10<=min(w,h)<=55 and 22<=max(w,h)<=140 and 1.6<=max(w,h)/min(w,h)<=5.5):continue
        poly=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
        if len(poly)!=5 or not cv2.isContourConvex(poly):continue
        if cv2.contourArea(contour)/(w*h)<.55:continue
        if any(inside([x+w/2,y+h/2],b) for b in excluded_boxes):continue
        axis=0 if w>h else 1;points=poly.reshape(-1,2);values=points[:,axis];across=points[:,1-axis]
        lo,hi=float(values.min()),float(values.max());span=hi-lo;cross=float((across.min()+across.max())/2)
        low_count=sum(abs(v-lo)<=span*.08 for v in values);high_count=sum(abs(v-hi)<=span*.08 for v in values)
        if sorted([low_count,high_count])!=[1,2]:continue
        tip_side=0 if low_count==1 else 1;tip=points[np.argmin(values) if tip_side==0 else np.argmax(values)]
        if abs(tip[1-axis]-cross)>max(2,(across.max()-across.min())*.12):continue
        ends=[[lo,cross],[hi,cross]] if axis==0 else [[cross,lo],[cross,hi]]
        branches=[branch(ends[0],axis,-1),branch(ends[1],axis,1)]
        if sum(branches)!=1:continue
        side=0 if branches[0] else 1;contact=ends[side]
        kind='net_input' if side==tip_side else 'net_output'
        labels=[];body=[float(x),float(y),float(x+w-1),float(y+h-1)]
        local=local_reader(body,axis,side) if local_reader is not None else []
        for t in [*items,*local]:
            text=t['text'].strip();b=t['bbox_xyxy'];cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
            if t.get('confidence',0)<.8 or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_+\-]{0,31}',text):continue
            if any(inside([cx,cy],box) for box in excluded_boxes):continue
            along=cx if axis==0 else cy
            if (side==0 and along<hi-3) or (side==1 and along>lo+3):continue
            distance=math.hypot(max(x-b[2],b[0]-(x+w-1),0),max(y-b[3],b[1]-(y+h-1),0))
            if distance>45:continue
            across_center=cy if axis==0 else cx
            text_span=b[3]-b[1] if axis==0 else b[2]-b[0]
            if abs(across_center-cross)>.45*text_span:continue
            labels.append((distance+abs((cy if axis==0 else cx)-cross)*.5,t))
        from port_text152 import complete_local_labels
        labels=complete_local_labels(labels,axis,side)
        labels.sort(key=lambda z:z[0])
        if not labels or (len(labels)>1 and labels[1][0]-labels[0][0]<3 and labels[0][1]['text']!=labels[1][1]['text']):continue
        label=labels[0][1]
        # Prefer outer contours among duplicate inner/outer holes.
        proposals.append(dict(type=kind,name=label['text'].strip(),value=label['text'].strip(),body_bbox_xyxy=body,
            axis='horizontal' if axis==0 else 'vertical',terminal_contacts_xy=[contact],terminal_outward_signs=[-1 if side==0 else 1],
            geometry_source='observed_pentagonal_port_single_wire_direction_and_ocr',reference_ocr_id=label['id'],label_evidence=dict(confidence=label['confidence'],bbox_xyxy=label['bbox_xyxy'],row_verified=True),
            direction_basis='wire_contact_at_pointed_end=input; wire_contact_at_flat_end=output',accepted=True))
    selected=[]
    for c in sorted(proposals,key=lambda c:-(c['body_bbox_xyxy'][2]-c['body_bbox_xyxy'][0])*(c['body_bbox_xyxy'][3]-c['body_bbox_xyxy'][1])):
        b=c['body_bbox_xyxy'];center=[(b[0]+b[2])/2,(b[1]+b[3])/2]
        if any(math.dist(center,[(p['body_bbox_xyxy'][0]+p['body_bbox_xyxy'][2])/2,(p['body_bbox_xyxy'][1]+p['body_bbox_xyxy'][3])/2])<8 for p in selected):continue
        selected.append(c)
    # A full-image OCR box can straddle adjacent port rows. Assign its text
    # to only the nearest aligned port; an unresolved tie supports neither.
    owned={}
    for c in selected:
        axis=0 if c['axis']=='horizontal' else 1
        b=c['label_evidence']['bbox_xyxy'];cross=c['terminal_contacts_xy'][0][1-axis]
        distance=abs((b[1]+b[3])/2-cross) if axis==0 else abs((b[0]+b[2])/2-cross)
        owned.setdefault(c['reference_ocr_id'],[]).append((distance,c))
    selected=[]
    for options in owned.values():
        options.sort(key=lambda item:item[0])
        if len(options)>1 and options[1][0]-options[0][0]<2:continue
        selected.append(options[0][1])
    seen=Counter()
    for c in sorted(selected,key=lambda c:(c['body_bbox_xyxy'][1],c['body_bbox_xyxy'][0])):
        text=c['value'];seen[text]+=1
        c['designator']=text+('_'+str(seen[text]) if seen[text]>1 else '')
        c['name']=text
    return selected


def recover_ports(image,items,excluded_boxes):
    """Bounded local OCR for geometry-confirmed labels, including rotated text."""
    from PIL import Image
    cache={};engine=None;attempts=0
    def local_reader(b,axis,side):
        nonlocal engine,attempts
        x,y,u,v=b
        if axis==0:region=[u+2,y-18,u+110,v+18] if side==0 else [x-110,y-18,x-2,v+18]
        else:region=[x-22,v+2,u+22,v+105] if side==0 else [x-22,y-105,u+22,y-2]
        region=[max(0,int(region[0])),max(0,int(region[1])),min(image.width,int(region[2])),min(image.height,int(region[3]))]
        key=tuple(region)
        if key in cache:return cache[key]
        if attempts>=24 or region[0]>=region[2] or region[1]>=region[3]:return []
        attempts+=1
        if engine is None:
            from rapidocr_onnxruntime import RapidOCR
            engine=RapidOCR()
        patch=image.crop(region);pw,ph=patch.size;found=[]
        for angle in ([0] if axis==0 else [90,270]):
            rotated=patch.rotate(angle,expand=True);scaled=rotated.resize((rotated.width*3,rotated.height*3),Image.Resampling.LANCZOS)
            result,_=engine(np.asarray(scaled))
            for j,(quad,text,confidence) in enumerate(result or []):
                if confidence<.85:continue
                ps=[]
                for xx,yy in quad:
                    xx,yy=xx/3,yy/3
                    if angle==90:xx,yy=pw-yy,xx
                    elif angle==270:xx,yy=yy,ph-xx
                    ps.append((xx+region[0],yy+region[1]))
                found.append(dict(id=f'port_ocr_{attempts}_{angle}_{j}',text=text,confidence=float(confidence),
                    bbox_xyxy=[min(p[0] for p in ps),min(p[1] for p in ps),max(p[0] for p in ps),max(p[1] for p in ps)],source='port_local_ocr'))
        cache[key]=found;return found
    return observe_ports(image,items,excluded_boxes,local_reader)
