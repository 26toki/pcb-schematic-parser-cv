"""Missing resistor values with a letter decimal marker, preserving observed text."""
import math
import re
from text_association import center,box_distance


def recover_resistor_value(component,label,items):
    if component['type']!='r' or label.get('value') is not None:return None
    ref=label.get('designator')
    if not ref or re.fullmatch(r'R(?:\d+|[A-Z]\d+|[a-z])',ref) is None:return None
    body=component['body_bbox_xyxy']
    anchors=[t for t in items if t['text'].strip()==ref and t.get('confidence',0)>=.8
             and box_distance(body,t['bbox_xyxy'])<=35]
    if not anchors:return None
    options=[]
    for t in items:
        if t.get('confidence',0)<.95:continue
        if re.fullmatch(r'(?:\d+[RrKkMm]\d+|\d+[Rr])[ΩΩ]?',t['text'].strip()) is None:continue
        tb=t['bbox_xyxy'];tx,ty=center(tb)
        if box_distance(body,tb)>30:continue
        for anchor in anchors:
            ab=anchor['bbox_xyxy'];ax,ay=center(ab)
            if not 3<ty-ay<=65 or abs(tx-ax)>max(8,min(15,(ab[2]-ab[0])*.5)):continue
            if any(a['text'].strip()!=ref and re.fullmatch(r'R(?:\d+|[A-Z]\d+|[a-z])',a['text'].strip())
                   and a.get('confidence',0)>=.8 and math.dist(center(a['bbox_xyxy']),[tx,ty])+5<math.dist([ax,ay],[tx,ty])
                   for a in items):continue
            score=box_distance(body,tb)+abs(tx-ax)+.1*(ty-ay);options.append((score,t,anchor))
    if not options:return None
    options.sort(key=lambda p:(p[0],-p[1]['confidence'],p[1]['text']))
    best=options[0]
    if any(t['text'].strip()!=best[1]['text'].strip() and score-best[0]<=5 for score,t,_ in options[1:]):return None
    return dict(value=best[1]['text'],source='aligned_explicit_resistor_letter_notation',
                value_ocr_id=best[1]['id'],value_confidence=best[1]['confidence'],value_bbox_xyxy=best[1]['bbox_xyxy'],
                reference_ocr_id=best[2]['id'],reference_bbox_xyxy=best[2]['bbox_xyxy'])
