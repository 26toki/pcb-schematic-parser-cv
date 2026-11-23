"""Read text beside an observed resistor without including its zigzag strokes."""
import math,re
import numpy as np
from PIL import Image
from text_association import box_distance,center

REFERENCE=re.compile(r'R\d+')
RESISTANCE=re.compile(r'(?:\d+(?:\.\d+)?[kKMm](?:[ΩΩ])?|\d+(?:\.\d+)?[ΩΩ]|\d+[RrKkMm]\d+|\d+(?:\.\d+)?[Rr])')

def read_side_label(image,component,geometries,engine):
    body=component['body_bbox_xyxy'];l,t,r,b=body
    depth=(r-l) if component['axis']=='vertical' else (b-t)
    reach=min(140,max(65,4*depth));pad=min(18,max(8,depth*.45))
    if component['axis']=='vertical':
        regions=[[r+2,t-pad,r+reach,b+1.3*depth],[l-reach,t-pad,l-2,b+1.3*depth]]
    else:
        regions=[[l-pad,t-reach,r+1.3*depth,t-2],[l-pad,b+2,r+1.3*depth,b+reach]]
    options=[]
    for region in regions:
        x0,y0,x1,y1=map(round,region);x0=max(0,x0);y0=max(0,y0);x1=min(image.width,x1);y1=min(image.height,y1)
        if x1-x0<12 or y1-y0<12:continue
        patch=image.crop((x0,y0,x1,y1));reads=[]
        for scale in [2,3]:
            try:raw,_=engine(np.asarray(patch.resize((patch.width*scale,patch.height*scale),Image.Resampling.LANCZOS)))
            except Exception:return None
            tokens=[]
            for polygon,text,confidence in raw or []:
                text=text.strip()
                if confidence<.97 or not(REFERENCE.fullmatch(text) or RESISTANCE.fullmatch(text)):continue
                xs=[p[0]/scale+x0 for p in polygon];ys=[p[1]/scale+y0 for p in polygon]
                tokens.append(dict(text=text,confidence=float(confidence),bbox_xyxy=[min(xs),min(ys),max(xs),max(ys)]))
            reads.append(tokens)
        matched=[]
        for a in reads[0]:
            peers=[v for v in reads[1] if a['text']==v['text'] and math.dist(center(a['bbox_xyxy']),center(v['bbox_xyxy']))<=3]
            if len(peers)==1:matched.append({**a,'confidence':min(a['confidence'],peers[0]['confidence'])})
        refs=[v for v in matched if REFERENCE.fullmatch(v['text'])]
        values=[v for v in matched if RESISTANCE.fullmatch(v['text'])]
        for ref in refs:
            for value in values:
                rb=ref['bbox_xyxy'];vb=value['bbox_xyxy'];font=max(6,rb[3]-rb[1]);rx,ry=center(rb);vx,vy=center(vb)
                # The adjacent reference/value must form one local text column.
                if not .5*font<vy-ry<2.5*font or abs(rb[0]-vb[0])>max(6,.5*font):continue
                if max(box_distance(body,rb),box_distance(body,vb))>max(40,1.5*depth):continue
                if any(box_distance(g['body_bbox_xyxy'],v['bbox_xyxy'])<=box_distance(body,v['bbox_xyxy'])+3
                    for g in geometries if g is not component for v in [ref,value]):continue
                options.append((ref,value))
    if len({(a['text'],b['text']) for a,b in options})!=1:return None
    a,b=options[0]
    return dict(detection_id='zigzag_local',type='r',designator=a['text'],value=b['text'],model=None,
        evidence={field:dict(ocr_id='local_zigzag_'+field,ocr_confidence=token['confidence'],bbox_xyxy=token['bbox_xyxy'],
            source='two_scale_OCR_outside_observed_zigzag_body') for field,token in [('designator',a),('value',b)]})
