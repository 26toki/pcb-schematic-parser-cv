"""Recover missing fields from concordant OCR and geometry, never substitute glyphs."""
import math,re
from text_association import center,box_distance

def contextual_reference(text,kind):
    prefix={'r':'R','c':'C','l':'L'}.get(kind)
    if prefix is None:return False
    return bool(re.fullmatch(prefix+r'_[A-Za-z][A-Za-z0-9_]*\d',text) or
        (kind=='r' and re.fullmatch(r'R[a-z]{2,10}',text)))

def _owned_by_component(token,component,candidates):
    own=box_distance(component['body_bbox_xyxy'],token['bbox_xyxy'])
    return not any(c['type']==component['type'] and math.dist(center(c['body_bbox_xyxy']),center(component['body_bbox_xyxy']))>5
        and box_distance(c['body_bbox_xyxy'],token['bbox_xyxy'])<=own+3 for c in candidates)

def recover_contextual_reference(component,label,full_items,local_items,candidates):
    if label.get('designator') is not None:return None
    if component.get('geometry_source') not in ('observed_rectangle_with_inline_leads','observed_beveled_rectangle_with_inline_leads',
        'observed_parallel_plates_with_inline_leads','observed_curved_and_straight_plates_with_inline_leads','observed_colored_coil_arcs_with_inline_leads'):return None
    body=component['body_bbox_xyxy'];options=[]
    for t in full_items:
        text=t['text'].strip()
        if not contextual_reference(text,component['type']) or t.get('confidence',0)<.95 or box_distance(body,t['bbox_xyxy'])>25:continue
        if not _owned_by_component(t,component,candidates):continue
        repeated=[u for u in local_items if u['text'].strip()==text and u.get('confidence',0)>=.95
            and math.dist(center(u['bbox_xyxy']),center(t['bbox_xyxy']))<=6]
        if not repeated:continue
        # An ordinary reference near the same body is a conflict, not a hint to overwrite it.
        if any(re.fullmatch({'r':'R','c':'C','l':'L'}[component['type']]+r'\d+',u['text'].strip()) and u.get('confidence',0)>=.9
            and box_distance(body,u['bbox_xyxy'])<=box_distance(body,t['bbox_xyxy'])+5 for u in full_items):continue
        options.append((text,t,repeated[0]))
    if len({o[0] for o in options})!=1:return None
    text,t,u=options[0]
    return dict(designator=text,source='matching_full_and_local_OCR_contextual_reference',
        full_reference_ocr_id=t['id'],local_reference_ocr_id=u['id'],reference_bbox_xyxy=t['bbox_xyxy'])

def recover_explicit_capacitance(component,label,full_items,local_items,candidates):
    if component['type']!='c' or label.get('value') is not None:return None
    ref=label.get('designator')
    if not ref or not(re.fullmatch(r'C\d+',ref) or contextual_reference(ref,'c')):return None
    body=component['body_bbox_xyxy']
    anchors=[t for t in full_items if t['text'].strip()==ref and t.get('confidence',0)>=.95
        and box_distance(body,t['bbox_xyxy'])<=45 and _owned_by_component(t,component,candidates)]
    options=[]
    for token in local_items:
        text=token['text'].strip();m=re.fullmatch(r'(\d+(?:\.\d+)?)[pnuμµm]?[Ff]',text)
        if not m or float(m[1])==0 or token.get('confidence',0)<.97:continue
        if not _owned_by_component(token,component,candidates):continue
        tx,ty=center(token['bbox_xyxy'])
        for anchor in anchors:
            ab=anchor['bbox_xyxy'];ax,ay=center(ab);font=max(8,ab[3]-ab[1])
            if not 3<ty-ay<=min(140,max(65,3.5*font)):continue
            if abs(tx-ax)>max(8,min(20,(ab[2]-ab[0])*.5)):continue
            if box_distance(body,token['bbox_xyxy'])>max(30,min(60,1.2*font)):continue
            options.append((text,token,anchor))
    if len({o[0] for o in options})!=1:return None
    text,t,a=options[0]
    return dict(value=text,source='explicit_local_capacitance_with_full_reference_and_unique_geometry_owner',
        reference_ocr_id=a['id'],value_ocr_id=t['id'],value_bbox_xyxy=t['bbox_xyxy'],value_confidence=t['confidence'])
