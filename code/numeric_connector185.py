"""Observed numeric connector names; no sequence completion or library inference."""
import re
from collections import Counter
from statistics import median

def numeric_pairs(box, items, reference, library_record):
    if library_record.get('pins') or not re.fullmatch(r'(?:FPC|USB|CN|J|H|P)\d+', reference or ''):
        return []
    x1,y1,x2,y2=box
    tokens=[t for t in items if re.fullmatch(r'\d{1,3}',t['text'].strip()) and t.get('confidence',0)>=.95]
    def mid(t):
        a,b,c,d=t['bbox_xyxy'];return ((a+c)/2,(b+d)/2)
    proposals=[]
    for side in ('left','right'):
        inside=[];outside=[]
        for t in tokens:
            x,y=mid(t)
            if not y1+3<y<y2-3:continue
            if side=='left':
                if x1-25<=x<x1:outside.append(t)
                elif x1<x<=min((x1+x2)/2,x1+25):inside.append(t)
            else:
                if x2<x<=x2+25:outside.append(t)
                elif max((x1+x2)/2,x2-25)<=x<x2:inside.append(t)
        for a in outside:
            matches=[b for b in inside if a['text'].strip()==b['text'].strip() and abs(mid(a)[1]-mid(b)[1])<=14]
            if len(matches)!=1:continue
            b=matches[0]
            proposals.append((side,a,b,mid(b)[1]-mid(a)[1]))
    counts=Counter(a['text'].strip() for _,a,_,_ in proposals)
    proposals=[p for p in proposals if counts[p[1]['text'].strip()]==1]
    result=[]
    for side in ('left','right'):
        group=[p for p in proposals if p[0]==side]
        if len(group)<3:continue
        offset=median(p[3] for p in group)
        if any(abs(p[3]-offset)>2 for p in group):continue
        ys=sorted(mid(p[2])[1] for p in group)
        if any(b-a<8 for a,b in zip(ys,ys[1:])):continue
        for _,a,b,delta in group:
            result.append(dict(number=a['text'].strip(),pinname=b['text'].strip(),side=side,
                number_ocr_id=a['id'],name_ocr_id=b['id'],row_difference_px=round(abs(delta),3),
                number_confidence=a['confidence'],name_confidence=b['confidence'],library_names=[],
                library_status='number_missing',pin_position=None,
                numeric_connector_evidence=dict(source='matching_observed_numeric_tokens',row_offset_px=offset,rows=len(group))))
    return result
