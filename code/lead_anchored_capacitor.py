"""Conservative fallback for named capacitors whose plate centroids are biased."""
import math,re
import numpy as np

def shared_lead_axis(mask,axis,a,b,box):
    from observe_passive_terminals import branch
    vertical=axis=='vertical';across=0 if vertical else 1;along=1-across
    lo=max(a['box'][across],b['box'][across]);hi=min(a['box'][across+2],b['box'][across+2])-1
    matches=[]
    for position in range(math.ceil(lo)+2,math.floor(hi)-1):
        contacts=[[position,box[1]],[position,box[3]]] if vertical else [[box[0],position],[box[2],position]]
        if branch(mask,axis,contacts[0],-1) and branch(mask,axis,contacts[1],1):matches.append(position)
    if not matches or len(matches)>5 or np.any(np.diff(matches)>1):return None
    position=float(np.median(matches))
    if any(abs(p['mid'][across]-position)>max(4,.2*min(a['length'],b['length'])) for p in [a,b]):return None
    first,second=sorted([a,b],key=lambda p:p['mid'][along])
    gap_start=first['box'][along+2];gap_end=second['box'][along]
    if gap_end<=gap_start:return None
    position=round(position)
    gap=mask[gap_start:gap_end,position-1:position+2] if vertical else mask[position-1:position+2,gap_start:gap_end].T
    if not len(gap) or not np.any(~gap.any(axis=1)):return None
    return dict(axis_position=float(position),support_positions=matches,source='unique_axis_supported_by_both_external_leads_and_empty_plate_gap')

def recover_lead_anchored_capacitor(mask,det,excluded):
    from observe_passive_terminals import capacitor_plates
    if det.get('type')!='c' or not re.fullmatch(r'C\d+',det.get('designator') or ''):return None
    confidence=det.get('evidence',{}).get('designator',{}).get('ocr_confidence',0)
    detection=det.get('detector_confidence',0)
    if confidence<.85 or detection<.25 or (confidence<.95 and detection<.7):return None
    b=det['body_bbox_xyxy'];cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
    if any(l<=cx<=r and t<=cy<=bottom for l,t,r,bottom in excluded):return None
    options=[]
    for axis in ['vertical','horizontal']:
        result=capacitor_plates(mask,det,axis_override=axis,min_length=8,lead_anchored=True)
        if result is not None:options.append(result)
    if len(options)!=1:return None
    return options[0]
