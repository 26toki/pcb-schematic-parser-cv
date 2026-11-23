"""Recover an existing weak voltage detection only with independent OCR and pixels."""
import math
import re
from text_association import center,box_distance
from attach_power_symbols import valid_label,inside_ic


def recover_low_voltage(det,items,ics,segments,mask,uf,groups):
    if det['class_name']!='v' or not .25<=det.get('confidence',0)<.5:return None
    raw=det.get('raw_bbox_xyxy',det['body_bbox_xyxy']);cx,cy=center(raw)
    if inside_ic([cx,cy],ics):return None
    labels=[]
    for t in items:
        if t.get('confidence',0)<.95 or inside_ic(center(t['bbox_xyxy']),ics):continue
        if not (valid_label(t['text'],'v') or re.fullmatch(r'(?:VBUS|V(?:CC|DD|SS)_?\d+\.\d+V)',t['text'].strip(),re.I)):continue
        if box_distance(raw,t['bbox_xyxy'])<=15:labels.append((math.dist([cx,cy],center(t['bbox_xyxy'])),t))
    labels.sort(key=lambda p:p[0])
    if not labels:return None
    if any(t['text'].strip().upper()!=labels[0][1]['text'].strip().upper() and d-labels[0][0]<=5 for d,t in labels[1:]):return None
    roots=[uf.find(i) for i in range(len(segments))];options={}
    def distance(point,s):
        if s['direction']=='vertical':return math.hypot(point[0]-s['axis'],max(s['start']-point[1],point[1]-s['end'],0))
        return math.hypot(point[1]-s['axis'],max(s['start']-point[0],point[0]-s['end'],0))
    for i,stem in enumerate(segments):
        length=stem['end']-stem['start']
        if stem['direction']!='vertical' or not 8<=length<=180 or abs(stem['axis']-cx)>3:continue
        x=round(stem['axis']);ys=range(round(stem['start']),round(stem['end'])+1)
        if not 1<=x<mask.shape[1]-1 or any(y<0 or y>=mask.shape[0] for y in ys):continue
        if sum(bool(mask[y,x-1:x+2].any()) for y in ys)/len(ys)<.85:continue
        for near,far in [(stem['start'],stem['end']),(stem['end'],stem['start'])]:
            if abs(near-cy)>8 or abs(far-cy)<abs(near-cy)+7:continue
            point=[stem['axis'],near];other=[stem['axis'],far]
            for j,bar in enumerate(segments):
                if bar['direction']!='horizontal' or not 6<=bar['end']-bar['start']<=40:continue
                if abs(bar['axis']-near)>2 or not bar['start']+2<=stem['axis']<=bar['end']-2:continue
                if abs((bar['start']+bar['end'])/2-stem['axis'])>3 or roots[j]!=roots[i]:continue
                wire_support=[s['id'] for k,s in enumerate(segments) if k not in (i,j) and roots[k]==roots[i]
                              and s['end']-s['start']>=20 and distance(other,s)<=8]
                contact_support=[p for p in groups.get(roots[i],{}).get('pins',[]) if p.get('kind') in ('ic','passive')]
                if not wire_support and not contact_support:continue
                boundary=min(math.dist(point,p) for p in [(cx,raw[1]),(cx,raw[3]),(raw[0],cy),(raw[2],cy)])
                if boundary>8:continue
                evidence=dict(source='weak_detector_with_confident_label_observed_bar_stem_and_connection',
                              detector_confidence=det['confidence'],ocr_confidence=labels[0][1]['confidence'],
                              ocr_bbox_xyxy=labels[0][1]['bbox_xyxy'],bar_segment_id=bar['id'],stem_segment_id=stem['id'],
                              supporting_wire_ids=wire_support,observed_contact_support_count=len(contact_support))
                options[(i,near)]=((boundary-.02*min(100,length),i,point),evidence)
    if len(options)!=1:return None
    terminal,evidence=next(iter(options.values()))
    return (labels[0][1],'v'),terminal,evidence
