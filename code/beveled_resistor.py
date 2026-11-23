"""One small raster bevel may hide a labelled rectangular resistor contour."""
import math,re
import cv2
import numpy as np

def rectangle_with_one_bevel(contour):
    poly=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True).reshape(-1,2)
    if len(poly)!=5:return None
    lengths=[float(np.linalg.norm(poly[(i+1)%5]-poly[i])) for i in range(5)]
    index=int(np.argmin(lengths));short=min(np.ptp(poly[:,0]),np.ptp(poly[:,1]))
    if lengths[index]>min(5,short*.45) or sorted(lengths)[1]<1.7*lengths[index]:return None
    merged=(poly[index].astype(float)+poly[(index+1)%5])/2
    quad=np.array([merged,*[poly[(index+i)%5] for i in [2,3,4]]],dtype=np.float32)
    if not cv2.isContourConvex(quad):return None
    directions=[]
    for p,q in zip(quad,np.roll(quad,-1,axis=0)):
        dx,dy=np.abs(p-q)
        if min(dx,dy)>3 or max(dx,dy)<5:return None
        directions.append('h' if dx>dy else 'v')
    if directions not in [['h','v','h','v'],['v','h','v','h']]:return None
    return dict(source='single_small_raster_bevel_on_labelled_rectangular_resistor',
        short_bevel_px=lengths[index],polygon_vertices=poly.tolist())

def recover_beveled_resistors(mask,detections,excluded_boxes):
    from observe_passive_terminals import branch
    eligible=[d for d in detections if d['type']=='r' and d.get('detector_confidence',0)>=.25
        and re.fullmatch(r'R(?:\d+|[A-Z]\d+|[a-z])',d.get('designator') or '')
        and d.get('value') is not None
        and d.get('evidence',{}).get('designator',{}).get('ocr_confidence',0)>=.95
        and d.get('evidence',{}).get('value',{}).get('ocr_confidence',0)>=.75]
    candidates=[];seen=set()
    for det in eligible:
        b=det['body_bbox_xyxy'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
        if any(x<=cx<=u and y<=cy<=v for x,y,u,v in excluded_boxes):continue
        l=max(0,math.floor(cx)-55);t=max(0,math.floor(cy)-55);r=min(mask.shape[1],math.ceil(cx)+56);bt=min(mask.shape[0],math.ceil(cy)+56)
        contours,_=cv2.findContours(mask[t:bt,l:r].astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for local in contours:
            contour=local+np.array([[[l,t]]],dtype=local.dtype)
            x,y,w,h=cv2.boundingRect(contour)
            if not (6<=min(w,h)<=30 and 16<=max(w,h)<=100 and 1.4<=max(w,h)/min(w,h)<=8):continue
            if cv2.contourArea(contour)/(w*h)<.65 or math.dist([cx,cy],[x+w/2,y+h/2])>12:continue
            evidence=rectangle_with_one_bevel(contour)
            if evidence is None:continue
            box=[x-1,y-1,x+w,y+h];axis='horizontal' if w>h else 'vertical'
            contacts=[[box[0],(box[1]+box[3])/2],[box[2],(box[1]+box[3])/2]] if axis=='horizontal' else [[(box[0]+box[2])/2,box[1]],[(box[0]+box[2])/2,box[3]]]
            if not branch(mask,axis,contacts[0],-1) or not branch(mask,axis,contacts[1],1):continue
            if tuple(box) in seen:continue
            seen.add(tuple(box));evidence.update(source_detection_id=det['detection_id'],reference=det['designator'],observed_value=det['value'])
            candidates.append(dict(type='r',body_bbox_xyxy=box,axis=axis,terminal_contacts_xy=contacts,
                geometry_source='observed_beveled_rectangle_with_inline_leads',beveled_rectangle_evidence=evidence))
    return candidates
