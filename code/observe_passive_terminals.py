"""Observe passive symbol bodies and two wire contacts from pixels; no GT inputs."""
import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from text_association import associate


def branch(mask,axis,point,sign):
    x,y=map(round,point);samples=[]
    for step in range(3,12):
        xx,yy=(x+sign*step,y) if axis=='horizontal' else (x,y+sign*step)
        if not (1<=xx<mask.shape[1]-1 and 1<=yy<mask.shape[0]-1):return False
        samples.append(mask[yy-1:yy+2,xx].any() if axis=='horizontal' else mask[yy,xx-1:xx+2].any())
    return sum(samples)/len(samples)>=.8


def diode_triangle_polygon(contour):
    """Allow one small raster bevel, while rejecting general quadrilaterals."""
    poly=cv2.approxPolyDP(contour,.045*cv2.arcLength(contour,True),True).reshape(-1,2)
    if len(poly)==3:return poly,False
    if len(poly)!=4:return None,False
    lengths=[float(np.linalg.norm(poly[(j+1)%4]-poly[j])) for j in range(4)]
    index=int(np.argmin(lengths));span=max(np.ptp(poly[:,0]),np.ptp(poly[:,1]))
    if lengths[index]>max(5,span/3) or sorted(lengths)[1]<2*lengths[index]:return None,False
    merged=(poly[index].astype(float)+poly[(index+1)%4])/2
    return np.array([merged,poly[(index+2)%4],poly[(index+3)%4]],dtype=float),True


def observed_diode_triangles(rgb,items,excluded_boxes):
    """Conservative color-symbol recovery: closed triangle, bar, two leads, OCR.

    No detections, library lookup, image name or annotation coordinates are used.
    Dense/ambiguous symbols remain unresolved. Internal terminal order is not A/K.
    """
    a=np.asarray(rgb).astype(np.int16);red=(a[:,:,0]-a[:,:,1]>45)&(a[:,:,0]-a[:,:,2]>45)&(a[:,:,0]>75)
    ink=np.asarray(Image.fromarray(a.astype(np.uint8)).convert('L'))<160
    contours,_=cv2.findContours(red.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    proposals=[]
    _,_,red_stats,red_centers=cv2.connectedComponentsWithStats(red.astype(np.uint8),8)
    arrow_parts=[]
    for stats,centroid in zip(red_stats[1:],red_centers[1:]):
        ax,ay,aw,ah,area=map(int,stats)
        if 5<=min(aw,ah) and max(aw,ah)<=50 and .5<=aw/ah<=2 and .15<=area/(aw*ah)<=.6:
            arrow_parts.append(([ax,ay,ax+aw,ay+ah],list(map(float,centroid))))
    def inside(p,b):return b[0]<=p[0]<=b[2] and b[1]<=p[1]<=b[3]
    text=[t for t in items if not any(inside([(t['bbox_xyxy'][0]+t['bbox_xyxy'][2])/2,(t['bbox_xyxy'][1]+t['bbox_xyxy'][3])/2],b) for b in excluded_boxes)]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour);poly,beveled=diode_triangle_polygon(contour)
        if poly is None or not (7<=min(w,h)<=45 and max(w,h)<=70 and cv2.contourArea(contour)>25):continue
        if any(inside([x+w/2,y+h/2],b) for b in excluded_boxes):continue
        # Axis is perpendicular to the nearly horizontal/vertical base.
        bases=[]
        for j in range(3):
            p,q=poly[j],poly[(j+1)%3];apex=poly[(j+2)%3]
            if abs(int(p[0]-q[0]))<=3 and abs(int(p[1]-q[1]))>=8:bases.append(('horizontal',p,q,apex))
            if abs(int(p[1]-q[1]))<=3 and abs(int(p[0]-q[0]))>=8:bases.append(('vertical',p,q,apex))
        if len(bases)!=1:continue
        axis,p,q,apex=bases[0];along=0 if axis=='horizontal' else 1;across=1-along
        center=float((p[across]+q[across])/2);tip=float(apex[along])
        if abs(float(apex[across])-center)>3:continue
        # A short clipped corner biases the base midpoint; retain the observed apex axis.
        if beveled:center=float(apex[across])
        span=abs(float(p[across]-q[across]))
        # Require a continuous transverse cathode stroke at the apex. The
        # triangle's own straight base must not count as cathode evidence.
        bars=[]
        for anchor in [tip]:
            for shift in range(-4,5):
                v=round(anchor+shift);coords=range(round(center-span/2),round(center+span/2)+1)
                samples=[red[z,v] if along==0 else red[v,z] for z in coords if 0<=v<red.shape[1-along] and 0<=z<red.shape[along]]
                if samples and sum(samples)/len(samples)>.85:bars.append(v)
        if not bars:continue
        lo=min(x if along==0 else y,min(bars))-1;hi=max(x+w-1 if along==0 else y+h-1,max(bars))+1
        contacts=[[lo,center],[hi,center]] if along==0 else [[center,lo],[center,hi]]
        if not (branch(ink,axis,contacts[0],-1) and branch(ink,axis,contacts[1],1)):continue
        for t in text:
            match=re.fullmatch(r'(LED|D)\d+',t['text'].strip(),re.I)
            if not match or t.get('confidence',0)<.8:continue
            tb=t['bbox_xyxy'];tx,ty=(tb[0]+tb[2])/2,(tb[1]+tb[3])/2;cx,cy=x+w/2,y+h/2
            distance=math.hypot(tx-cx,ty-cy)
            if distance>95:continue
            # Vertical EasyEDA diodes normally place the reference to the right;
            # reject labels on the preceding device in a compact LED bank.
            if along==1 and tb[0]<cx+3:continue
            typ='led' if match[1].upper()=='LED' else 'd';cid=t['text'].strip().upper()
            models=[]
            for token in text:
                s=token['text'].strip();b=token['bbox_xyxy'];mx,my=(b[0]+b[2])/2,(b[1]+b[3])/2
                if re.fullmatch(r'(?:LED|[RCLDUJ])\d+',s,re.I) or not(re.search('[A-Za-z]',s) and re.search('[0-9]',s)):continue
                if not re.fullmatch(r'[A-Za-z0-9_.+\-]+',s) or len(s)<4:continue
                if math.hypot(mx-tx,my-ty)>65 or my<ty-8:continue
                models.append((math.hypot(mx-tx,my-ty),s))
            name=min(models)[1] if models else cid
            core=[lo,center-span/2-1,hi,center+span/2+1] if along==0 else [center-span/2-1,lo,center+span/2+1,hi]
            b=core.copy()
            if typ=='led':
                b[across]-=span*.5;b[across+2]+=span*.5
                # Include the two detached light arrows when their actual ink
                # supplies consistent same-side evidence. A symmetric triangle
                # expansion misses the along-axis extent of horizontal LEDs.
                core_center=[(core[0]+core[2])/2,(core[1]+core[3])/2]
                nearby=[]
                for ab,ac in arrow_parts:
                    da,dc=ac[along]-core_center[along],ac[across]-core_center[across]
                    if max(ab[2]-ab[0],ab[3]-ab[1])>1.5*span:continue
                    if .25*span<=abs(dc)<=2*span and abs(da)<=2*span:nearby.append((ab,ac,da,dc))
                pairs=[]
                for j,first in enumerate(nearby):
                    for second in nearby[j+1:]:
                        arrow_distance=math.dist(first[1],second[1])
                        if first[2]*second[2]>0 and first[3]*second[3]>0 and 4<=arrow_distance<=1.5*span:
                            pairs.append((math.dist(first[1],core_center)+math.dist(second[1],core_center),first[0],second[0]))
                if pairs:
                    _,one,two=min(pairs,key=lambda p:p[0]);b=[min(b[0],one[0],two[0]),min(b[1],one[1],two[1]),max(b[2],one[2],two[2]),max(b[3],one[3],two[3])]
            proposals.append((distance,cid,dict(type=typ,designator=cid,name=name,value=None,body_bbox_xyxy=b,wire_mask_bbox_xyxy=core,
                axis=axis,terminal_contacts_xy=contacts,geometry_source='observed_red_triangle_bar_two_leads_and_reference',
                reference_ocr_id=t['id'],triangle_bbox_xyxy=[x,y,x+w-1,y+h-1],
                **({'triangle_raster_bevel_recovery':True} if beveled else {}))))
    selected=[];used=set()
    for _,cid,c in sorted(proposals,key=lambda p:p[0]):
        center=[(c['triangle_bbox_xyxy'][0]+c['triangle_bbox_xyxy'][2])/2,(c['triangle_bbox_xyxy'][1]+c['triangle_bbox_xyxy'][3])/2]
        if cid in used or any(math.dist(center,[(u['triangle_bbox_xyxy'][0]+u['triangle_bbox_xyxy'][2])/2,(u['triangle_bbox_xyxy'][1]+u['triangle_bbox_xyxy'][3])/2])<5 for u in selected):continue
        selected.append(c);used.add(cid)
    return selected


def recover_led_reference_text(image,items,engine,limit=12):
    """Recover a missing leading L by re-reading pixels at two scales.

    ED<number> only triggers inspection. The result is accepted only when both
    OCR reads explicitly return the same full LED<number>, never by prefixing
    characters to the original OCR text. Symbol geometry is checked separately.
    """
    existing={t['text'].strip().upper() for t in items};recovered=[];attempts=[]
    candidates=[t for t in items if re.fullmatch(r'ED\d+',t['text'].strip(),re.I) and t['confidence']>=.9]
    for original in candidates[:limit]:
        expected='L'+original['text'].strip().upper()
        if expected in existing:continue
        x,y,u,v=original['bbox_xyxy'];left=max(0,math.floor(x)-20);top=max(0,math.floor(y)-8)
        right=min(image.width,math.ceil(u)+12);bottom=min(image.height,math.ceil(v)+6)
        crop=image.crop((left,top,right,bottom)).convert('RGB');votes=[]
        for scale in (3,4):
            raw,_=engine(np.asarray(crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.LANCZOS)))
            matches=[]
            for quad,text,confidence in raw or []:
                if text.strip().upper()!=expected or confidence<.95:continue
                xs=[float(p[0])/scale+left for p in quad];ys=[float(p[1])/scale+top for p in quad]
                matches.append(dict(text=text.strip().upper(),confidence=float(confidence),bbox_xyxy=[min(xs),min(ys),max(xs),max(ys)]))
            votes.append(max(matches,key=lambda t:t['confidence']) if matches else None)
        attempts.append(dict(source_ocr_id=original['id'],crop_xyxy=[left,top,right,bottom],votes=votes))
        if any(v is None for v in votes):continue
        record=dict(votes[0],id='led_reference_'+original['id'],source='local_led_reference_two_scale_consensus',
                    original_text=original['text'],confidence=min(v['confidence'] for v in votes))
        recovered.append(record);existing.add(expected)
    return recovered,attempts


def merge_rescued_label(label,local):
    """Fill missing fields only when local OCR agrees with an existing identity."""
    result=dict(label)
    # A zero-valued capacitance cannot establish a missing physical component
    # value. Keep it unresolved rather than guess the missing leading digit.
    value=local.get('value') or ''
    number=re.fullmatch(r'(\d+(?:\.\d+)?)[pnuμµm]?[Ff]',value)
    if (local.get('designator') or '').startswith('C') and number and float(number[1])==0:return result
    if label['designator'] is not None and local['designator']!=label['designator']:return result
    if label['value'] is not None and local['value'] is not None and local['value']!=label['value']:return result
    if local['designator'] is None:return result
    for field in ('designator','value'):
        if result[field] is None and local[field] is not None:result[field]=local[field]
    return result


def resistor_rectangles(mask,*,max_short=30,max_long=100):
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if not (6<=min(w,h)<=max_short and 16<=max(w,h)<=max_long and 1.4<=max(w,h)/min(w,h)<=8):continue
        if cv2.contourArea(contour)/(w*h)<.65:continue
        poly=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
        if len(poly)!=4:continue
        # Hole contour follows the inner edge of a rectangular symbol stroke.
        box=[x-1,y-1,x+w,y+h];axis='horizontal' if w>h else 'vertical'
        terminals=[[box[0],(box[1]+box[3])/2],[box[2],(box[1]+box[3])/2]] if axis=='horizontal' else [[(box[0]+box[2])/2,box[1]],[(box[0]+box[2])/2,box[3]]]
        if branch(mask,axis,terminals[0],-1) and branch(mask,axis,terminals[1],1):
            candidates.append(dict(type='r',body_bbox_xyxy=box,axis=axis,terminal_contacts_xy=terminals,geometry_source='observed_rectangle_with_inline_leads'))
    return candidates


def coil_components(ink,mask,reference=None):
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink,8);result=[]
    for index in range(1,count):
        x,y,w,h,area=map(int,stats[index]);horizontal=w>=h;long=max(w,h);short=min(w,h)
        if not (24<=long<=180 and 6<=short<=40 and 2.3<=long/short<=18):continue
        patch=(labels[y:y+h,x:x+w]==index)
        if reference is not None:
            patch &= reference[y:y+h,x:x+w].astype(bool)
            ys,xs=np.where(patch)
            if not len(xs):continue
            xoff,yoff=int(xs.min()),int(ys.min());patch=patch[yoff:int(ys.max())+1,xoff:int(xs.max())+1]
            x+=xoff;y+=yoff;h,w=patch.shape
        view=patch if horizontal else patch.T
        height,width=view.shape
        left=np.where(view[:,:2])[0];right=np.where(view[:,-2:])[0]
        if not len(left) or not len(right) or abs(np.median(left)-np.median(right))>4:continue
        estimated=(float(np.median(left))+float(np.median(right)))/2
        choices=[]
        for row in range(round(estimated)-4,round(estimated)+5):
            p=[x-1,y+row] if horizontal else [x+row,y-1]
            q=[x+w,y+row] if horizontal else [x+row,y+h]
            axis='horizontal' if horizontal else 'vertical'
            perpendicular='vertical' if horizontal else 'horizontal';outward=1 if row>=height/2 else -1
            left_inline=branch(mask,axis,p,-1);right_inline=branch(mask,axis,q,1)
            left_supported=left_inline or branch(mask,perpendicular,p,outward)
            right_supported=right_inline or branch(mask,perpendicular,q,outward)
            if left_supported and right_supported:
                # Ellipse end strokes may be several pixels thick. Locate the
                # contact on the observed lead center, rather than the first
                # row that happens to touch its stroke margin.
                alignment=[]
                for point,sign,inline in ((p,-1,left_inline),(q,1,right_inline)):
                    if not inline:continue
                    offsets=[]
                    for step in range(3,12):
                        px,py=point
                        px,py=(px+sign*step,py) if horizontal else (px,py+sign*step)
                        values=[]
                        for delta in range(-3,4):
                            xx,yy=(px,py+delta) if horizontal else (px+delta,py)
                            if 0<=xx<mask.shape[1] and 0<=yy<mask.shape[0] and mask[yy,xx]:values.append(delta)
                        if values:offsets.append(float(np.mean(values)))
                    if offsets:alignment.append(abs(float(np.median(offsets))))
                choices.append((float(np.mean(alignment)) if alignment else 0,abs(row-estimated),row,p,q))
        if not choices:continue
        _,_,baseline,p,q=min(choices)
        if min(abs(baseline),abs(height-1-baseline))>.35*height:continue
        above=baseline>=height/2;profile=[]
        for col in range(width):
            ys=np.flatnonzero(view[:,col])
            if not len(ys):profile.append(0);continue
            profile.append(max(0,baseline-int(ys.min())) if above else max(0,int(ys.max())-baseline))
        profile=np.convolve(profile,np.ones(3)/3,mode='same')
        peak=float(profile.max())
        if peak<4:continue
        high=profile>=max(3,.65*peak)
        edges=np.diff(np.r_[False,high,False].astype(np.int8));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        centers=[(a+b-1)/2 for a,b in zip(starts,ends) if b-a>=2]
        if not 2<=len(centers)<=8:continue
        spacing=np.diff(centers)
        if min(spacing)<4 or max(spacing)/min(spacing)>2.8 or centers[-1]-centers[0]<.4*width:continue
        result.append(dict(type='l',body_bbox_xyxy=[x,y,x+w-1,y+h-1],axis=axis,terminal_contacts_xy=[p,q],
                           geometry_source='observed_colored_coil_arcs_with_inline_leads',
                           coil_evidence=dict(repeated_arcs=len(centers),arc_depth_px=peak,one_sided=True,lead_axis_verified=True)))
    return result


def observed_coils(rgb,mask):
    """Repeated one-sided red arcs with observed wires; straight joins may be removed for isolation."""
    r,g,b=rgb.astype(np.int16).transpose(2,0,1)
    ink=((r-g>35)&(r-b>35)&(r>70)).astype(np.uint8)
    straight=np.zeros_like(ink)
    for kernel in ((14,1),(1,14)):
        straight |= cv2.morphologyEx(ink,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,kernel))
    curved=ink & (1-straight)
    joined=cv2.dilate(curved,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    result=[]
    for item in coil_components(ink,mask)+coil_components(joined,mask,ink):
        b=item['body_bbox_xyxy'];center=[(b[0]+b[2])/2,(b[1]+b[3])/2]
        if any(math.dist(center,[(c['body_bbox_xyxy'][0]+c['body_bbox_xyxy'][2])/2,(c['body_bbox_xyxy'][1]+c['body_bbox_xyxy'][3])/2])<8 for c in result):continue
        result.append(item)
    return result


def curved_plate_evidence(mask,axis,short,long,spacing):
    """A shortened horizontal-opening response must belong to an observed bowed stroke."""
    along=1 if axis=='vertical' else 0;across=1-along
    sign=1 if short['mid'][along]>long['mid'][along] else -1
    near=max(1,min(3,int(spacing/2)-1));far=min(10,max(4,round(long['length']*.3)))
    low,high=(-near,far) if sign>0 else (-far,near)
    offsets=[];start=long['box'][across];end=long['box'][across+2]
    mid=short['mid'][along]
    for cross in range(start,end):
        if abs(cross-short['mid'][across])<=2:continue  # Do not mistake the axial lead for the plate.
        ink=[]
        for value in range(math.floor(mid+low),math.ceil(mid+high)+1):
            x,y=(cross,value) if axis=='vertical' else (value,cross)
            if 0<=x<mask.shape[1] and 0<=y<mask.shape[0] and mask[y,x]:ink.append(value)
        if ink:offsets.append((cross,min(ink,key=lambda v:abs(v-mid))-mid))
    expected=sum(abs(c-short['mid'][across])>2 for c in range(start,end))
    if len(offsets)<.8*max(1,expected):return None
    span=max(1,end-start)
    left=[v for c,v in offsets if c<start+.25*span];right=[v for c,v in offsets if c>=start+.75*span]
    center=[v for c,v in offsets if start+.3*span<=c<start+.7*span]
    if not left or not right or not center:return None
    center=float(np.median(center));dl=sign*(float(np.median(left))-center);dr=sign*(float(np.median(right))-center)
    if min(dl,dr)<1 or abs(dl-dr)>3:return None
    return dict(status='observed_bowed_plate_across_straight_plate_span',coverage=len(offsets)/max(1,expected),edge_bow_px=[dl,dr])


def capacitor_plates(mask,det,*,axis_override=None,min_length=12,lead_anchored=False):
    b=det['body_bbox_xyxy'];cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
    axis=axis_override or ('vertical' if b[3]-b[1]>=b[2]-b[0] else 'horizontal')
    left,top=max(0,round(cx)-45),max(0,round(cy)-45)
    patch=mask[top:min(mask.shape[0],round(cy)+46),left:min(mask.shape[1],round(cx)+46)].astype(np.uint8)
    kernel=(min_length,1) if axis=='vertical' else (1,min_length)
    opened=cv2.morphologyEx(patch,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,kernel))
    count,labels,stats,centers=cv2.connectedComponentsWithStats(opened,8);plates=[]
    for i in range(1,count):
        x,y,w,h,_=stats[i];length=w if axis=='vertical' else h;thickness=h if axis=='vertical' else w
        if not min_length<=length<=80 or thickness>5:continue
        mid=[float(centers[i][0])+left,float(centers[i][1])+top]
        if abs(mid[0]-cx)>12 or abs(mid[1]-cy)>20:continue
        plates.append(dict(mid=mid,box=[int(x)+left,int(y)+top,int(x+w)+left,int(y+h)+top],length=int(length)))
    proposals=[]
    for i,a in enumerate(plates):
        for c in plates[i+1:]:
            coord=1 if axis=='vertical' else 0;other=1-coord
            spacing=abs(a['mid'][coord]-c['mid'][coord])
            if not 4<=spacing<=18 or abs(a['mid'][other]-c['mid'][other])>3:continue
            ratio=min(a['length'],c['length'])/max(a['length'],c['length']);curve=None
            if ratio<.7:
                if ratio<.35:continue
                short,long=sorted((a,c),key=lambda p:p['length'])
                curve=curved_plate_evidence(mask,axis,short,long,spacing)
                if curve is None:continue
            box=[min(a['box'][0],c['box'][0]),min(a['box'][1],c['box'][1]),max(a['box'][2],c['box'][2])-1,max(a['box'][3],c['box'][3])-1]
            center=[(box[0]+box[2])/2,(box[1]+box[3])/2]
            lead_evidence=None
            if lead_anchored:
                from lead_anchored_capacitor import shared_lead_axis
                lead_evidence=shared_lead_axis(mask,axis,a,c,box)
                if lead_evidence is None:continue
                center[other]=lead_evidence['axis_position']
            terminals=[[center[0],box[1]],[center[0],box[3]]] if axis=='vertical' else [[box[0],center[1]],[box[2],center[1]]]
            if branch(mask,axis,terminals[0],-1) and branch(mask,axis,terminals[1],1):
                geometry=dict(type='c',body_bbox_xyxy=box,axis=axis,terminal_contacts_xy=terminals,
                              geometry_source='observed_curved_and_straight_plates_with_inline_leads' if curve else 'observed_parallel_plates_with_inline_leads')
                if curve:geometry['curved_plate_evidence']=curve
                if lead_evidence:geometry['inline_lead_evidence']=lead_evidence
                proposals.append((math.dist(center,[cx,cy]),geometry))
    if lead_anchored and len(proposals)>1:
        centers=[[(g['body_bbox_xyxy'][0]+g['body_bbox_xyxy'][2])/2,(g['body_bbox_xyxy'][1]+g['body_bbox_xyxy'][3])/2] for _,g in proposals]
        if any(math.dist(centers[0],p)>4 for p in centers[1:]):return None
    return min(proposals,key=lambda x:x[0])[1] if proposals else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('associations','ocr','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--rescue-ocr',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    source=json.loads(a.associations.read_text(encoding='utf-8'));image=Image.open(source['image']).convert('RGB')
    mask=np.asarray(image.convert('L'))<160;items=json.loads(a.ocr.read_text(encoding='utf-8'))['items']
    engine=None;reference_attempts=[]
    if a.rescue_ocr and any(re.fullmatch(r'ED\d+',t['text'].strip(),re.I) and t['confidence']>=.9 for t in items):
        from rapidocr_onnxruntime import RapidOCR
        engine=RapidOCR();recovered,reference_attempts=recover_led_reference_text(image,items,engine)
        items=items+recovered
    detections=[p for p in source['predictions'] if p['type'] in ('r','c','l')]
    candidates=resistor_rectangles(mask)+observed_coils(np.asarray(image),mask)
    excluded=[d['body_bbox_xyxy'] for d in source['predictions'] if d['type']=='box' and d.get('designator')]
    from beveled_resistor import recover_beveled_resistors
    candidates+=recover_beveled_resistors(mask,detections,excluded)
    from zigzag_resistors import zigzag_geometries
    from merged_resistor_text import label_zigzag_geometries
    zigzags=zigzag_geometries(mask,excluded)
    if zigzags:
        if a.rescue_ocr and engine is None:
            from rapidocr_onnxruntime import RapidOCR
            engine=RapidOCR()
        labelled_zigzags,zigzag_text_stats=label_zigzag_geometries(image,zigzags,items,engine)
        candidates+=labelled_zigzags
    candidates+=observed_diode_triangles(np.asarray(image),items,excluded)
    for det in detections:
        if det['type']=='c':
            observed=capacitor_plates(mask,det)
            if observed is None:
                from lead_anchored_capacitor import recover_lead_anchored_capacitor
                observed=recover_lead_anchored_capacitor(mask,det,excluded)
            if observed:candidates.append(observed)
        elif det['type']=='r' and not any(math.dist([(p['body_bbox_xyxy'][0]+p['body_bbox_xyxy'][2])/2,(p['body_bbox_xyxy'][1]+p['body_bbox_xyxy'][3])/2],[(det['body_bbox_xyxy'][0]+det['body_bbox_xyxy'][2])/2,(det['body_bbox_xyxy'][1]+det['body_bbox_xyxy'][3])/2])<12 for p in candidates if p['type']=='r'):
            # Zigzag resistors do not form a closed rectangle. Keep detector geometry as a hypothesis.
            b=det['body_bbox_xyxy'];cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2;axis='horizontal' if b[2]-b[0]>b[3]-b[1] else 'vertical'
            terminals=[[b[0],cy],[b[2],cy]] if axis=='horizontal' else [[cx,b[1]],[cx,b[3]]]
            if branch(mask,axis,terminals[0],-1) and branch(mask,axis,terminals[1],1):candidates.append(dict(type='r',body_bbox_xyxy=b,axis=axis,terminal_contacts_xy=terminals,geometry_source='detector_body_hypothesis_with_inline_leads'))
    unique=[]
    for c in candidates:
        b=c['body_bbox_xyxy'];center=[(b[0]+b[2])/2,(b[1]+b[3])/2]
        if not any(u['type']==c['type'] and math.dist(center,[(u['body_bbox_xyxy'][0]+u['body_bbox_xyxy'][2])/2,(u['body_bbox_xyxy'][1]+u['body_bbox_xyxy'][3])/2])<5 for u in unique):unique.append(c)
    if a.rescue_ocr:
        from rapidocr_onnxruntime import RapidOCR
        from read_schematic_text import recognize
        if engine is None:engine=RapidOCR()
    results=[];canvas=image.copy();draw=ImageDraw.Draw(canvas);text_view_attempts=[]
    for index,c in enumerate(unique):
        c['id']=f'passive_{index:04d}';b=c['body_bbox_xyxy'];center=[(b[0]+b[2])/2,(b[1]+b[3])/2]
        near=[d for d in detections if d['type']==c['type'] and math.dist(center,[(d['body_bbox_xyxy'][0]+d['body_bbox_xyxy'][2])/2,(d['body_bbox_xyxy'][1]+d['body_bbox_xyxy'][3])/2])<20]
        det=dict(id=c['id'],class_name=c['type'],body_bbox_xyxy=b)
        if c.get('image_text_label'):label=dict(c['image_text_label'])
        elif c['type'] in ('d','led'):label=dict(designator=c['designator'],value=None)
        else:
            pred,_,_=associate([det],items,{},70,20,allow_extended_resistor_refs=True);label=pred[0]
        if near and not c.get('image_text_label'):
            best=min(near,key=lambda d:math.dist(center,[(d['body_bbox_xyxy'][0]+d['body_bbox_xyxy'][2])/2,(d['body_bbox_xyxy'][1]+d['body_bbox_xyxy'][3])/2]))
            label['designator']=best['designator'];label['value']=best['value'];c['source_detection_id']=best['detection_id']
        local=[]
        if engine is not None and c['type'] not in ('d','led') and (label['designator'] is None or label['value'] is None):
            region=[max(0,math.floor(b[0]-55)),max(0,math.floor(b[1]-55)),min(image.width,math.ceil(b[2]+56)),min(image.height,math.ceil(b[3]+56))]
            folder=a.output/'rescue_ocr'/c['id'];recognize(engine,image,region,3,folder)
            local=json.loads((folder/'ocr.json').read_text(encoding='utf-8'))['items'];lp,_,_=associate([det],local,{},70,20,allow_extended_resistor_refs=True)
            merged=merge_rescued_label(label,lp[0])
            if (merged['designator'],merged['value'])!=(label['designator'],label['value']):
                label=merged;c['label_source']='local_rescue_ocr'
        from contextual_passive_text import recover_contextual_reference,recover_explicit_capacitance
        context_ref=recover_contextual_reference(c,label,items,local,unique)
        if context_ref is not None:
            label['designator']=context_ref['designator'];c['contextual_reference_recovery']=context_ref
        explicit_cap=recover_explicit_capacitance(c,label,items,local,unique)
        if explicit_cap is not None:
            label['value']=explicit_cap['value'];c['explicit_capacitance_recovery']=explicit_cap
        from capacitance_shorthand import recover_capacitance_value
        recovered=recover_capacitance_value(c,label,items+local)
        if recovered is not None:
            label['value']=recovered['value'];c['capacitance_value_recovery']=recovered
        from resistor_shorthand import recover_resistor_value
        resistance=recover_resistor_value(c,label,items+local)
        if resistance is not None:
            label['value']=resistance['value'];c['resistance_value_recovery']=resistance
        from resistor_text_views import needs_text_views,recover_text_views
        if engine is not None and len(text_view_attempts)<12 and needs_text_views(c,label):
            repaired,evidence=recover_text_views(image,c,unique,label,engine)
            text_view_attempts.append(dict(candidate_id=c['id'],**evidence))
            if repaired is not None:
                label.update(repaired);c['resistor_text_view_recovery']=evidence
        c.update(designator=label['designator'],value=label['value'],terminal_numbers_are_internal_order=True)
        # Unnamed geometric candidates remain proposals and are not used to cut a wire.
        c['accepted']=label['designator'] is not None and (label['value'] is not None or c['type'] in ('d','led'))
        results.append(c);draw.rectangle(b,outline='blue' if c['accepted'] else 'orange',width=2)
        for x,y in c['terminal_contacts_xy']:draw.ellipse((x-3,y-3,x+3,y+3),outline='red',width=2)
    from large_resistor161 import recover as recover_large_resistors
    large_added,large_events=recover_large_resistors(image,mask,results,items,engine,a.output/'large_rescue',resistor_rectangles,recognize,excluded) if engine is not None else ([],[])
    results+=large_added
    from observe_network_ports import recover_ports
    used_designators={c['designator'] for c in results if c['accepted']}
    for index,port in enumerate(recover_ports(image,items,excluded)):
        if port['designator'] in used_designators:continue
        port.update(id=f'net_port_{index:04d}',terminal_numbers_are_internal_order=True)
        results.append(port)
    repeated=Counter(c['designator'] for c in results if c['accepted'])
    for c in results:
        if c['accepted'] and repeated[c['designator']]>1:
            c['accepted']=False;c['review_reason']='same_designator_at_multiple_candidate_geometries'
    canvas=image.copy();draw=ImageDraw.Draw(canvas)
    for c in results:
        draw.rectangle(c['body_bbox_xyxy'],outline='blue' if c['accepted'] else 'orange',width=2)
        for x,y in c['terminal_contacts_xy']:draw.ellipse((x-3,y-3,x+3,y+3),outline='red',width=2)
    summary=dict(candidates=len(results),accepted=sum(c['accepted'] for c in results),
                 geometry_rescued_accepted=sum(c['accepted'] and 'source_detection_id' not in c for c in results))
    report=dict(image=source['image'],summary=summary,components=results,large_resistor_recovery=large_events,led_reference_rescue=reference_attempts,
                resistor_text_view_attempts=text_view_attempts,
                limitations='Image/YOLO/OCR only, no GT. Axis-aligned r/c/l candidates and conservatively gated colored d/led symbols. Contacts are symbol-body contacts, not final pin tips. Unnamed proposals are retained, not used for masking. Internal terminal order is not official pin numbering or diode polarity.')
    (a.output/'passives.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    canvas.save(a.output/'passive_contacts.png');print(json.dumps(summary))


if __name__=='__main__':main()
