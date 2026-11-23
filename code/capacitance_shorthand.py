"""Conservative missing-value recovery for observed capacitors; preserve image text."""
import math
import re
from text_association import center,box_distance


def recover_capacitance_value(component,label,items):
    """Only explicit C-number anchors and nearby aligned, high-confidence short values.

    Does not classify generic numbers as values, rewrite units, replace an existing
    value or supply a value to another component category. Four-digit packages
    such as 0603 and ambiguous OCR substitutions are deliberately unsupported.
    """
    if component['type']!='c' or label.get('value') is not None:
        return None
    ref=label.get('designator')
    if not ref or re.fullmatch(r'C\d+',ref) is None:
        return None
    body=component['body_bbox_xyxy']
    anchors=[t for t in items if t['text'].strip()==ref and t.get('confidence',0)>=.8
             and box_distance(body,t['bbox_xyxy'])<=40]
    if not anchors:return None
    options=[]
    for token in items:
        text=token['text'].strip()
        if token.get('confidence',0)<.95:continue
        if re.fullmatch(r'(?:[1-9]\d{2}|\d+(?:\.\d+)?[pnuμµ])',text) is None:continue
        box=token['bbox_xyxy'];tx,ty=center(box)
        if box_distance(body,box)>35:continue
        for anchor in anchors:
            ab=anchor['bbox_xyxy'];ax,ay=center(ab)
            # Capacitor reference and value must form a nearby vertical text column.
            if not 3<ty-ay<=65 or abs(tx-ax)>max(8,min(15,(ab[2]-ab[0])*.5)):continue
            if any(t['text'].strip()!=ref and re.fullmatch(r'C\d+',t['text'].strip())
                   and t.get('confidence',0)>=.8
                   and math.dist(center(t['bbox_xyxy']),[tx,ty])+5<math.dist([ax,ay],[tx,ty])
                   for t in items):continue
            score=box_distance(body,box)+abs(tx-ax)+.1*(ty-ay)
            options.append((score,token,anchor))
    if not options:return None
    options.sort(key=lambda p:(p[0],-p[1]['confidence'],p[1]['text']))
    best=options[0]
    if any(t['text'].strip()!=best[1]['text'].strip() and score-best[0]<=5 for score,t,_ in options[1:]):
        return None
    return dict(value=best[1]['text'],source='aligned_explicit_capacitance_shorthand',
                value_ocr_id=best[1]['id'],value_confidence=best[1]['confidence'],
                value_bbox_xyxy=best[1]['bbox_xyxy'],reference_ocr_id=best[2]['id'],
                reference_bbox_xyxy=best[2]['bbox_xyxy'])
