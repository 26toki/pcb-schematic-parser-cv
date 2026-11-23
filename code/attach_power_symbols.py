"""Attach detected ground/power symbols and OCR text to existing physical wire groups."""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import numpy as np
from PIL import Image,ImageDraw
from text_association import box_distance,center
from trace_ic_wire_networks import connect_segments


def valid_label(text,kind):
    t=text.strip().upper()
    if kind=='gnd':return bool(re.fullmatch(r'(?:[ADP]?GND[A-Z0-9_]*|GROUND)',t))
    return bool(re.fullmatch(r'(?:[+\-]?\d+(?:\.\d+)?V|[+\-]?\d+V\d+|VCC[A-Z0-9_]*|VDD[A-Z0-9_]*|VSS[A-Z0-9_]*|VBAT)',t))


def inside_ic(point,ics):
    return any(b[0]<=point[0]<=b[2] and b[1]<=point[1]<=b[3] for b in (ic['body_bbox_xyxy'] for ic in ics))


def choose_label(det,items,ics):
    """Recover a detector category conflict only from a nearby explicit OCR label."""
    raw=det.get('raw_bbox_xyxy',det['body_bbox_xyxy']);kind=det['class_name']
    candidates=[t for t in items if t['confidence']>=.8 and not inside_ic(center(t['bbox_xyxy']),ics)
                and box_distance(t['bbox_xyxy'],raw)<=40]
    matching=[t for t in candidates if valid_label(t['text'],kind)]
    if matching:return min(matching,key=lambda t:math.dist(center(t['bbox_xyxy']),center(raw))),kind
    other='v' if kind=='gnd' else 'gnd'
    alternative=[t for t in candidates if t['confidence']>=.9 and valid_label(t['text'],other)
                 and box_distance(t['bbox_xyxy'],raw)<=20]
    if not alternative:return None
    alternative.sort(key=lambda t:math.dist(center(t['bbox_xyxy']),center(raw)))
    if len(alternative)>1 and alternative[0]['text']!=alternative[1]['text']:
        da=math.dist(center(alternative[0]['bbox_xyxy']),center(raw))
        db=math.dist(center(alternative[1]['bbox_xyxy']),center(raw))
        if db-da<=5:return None
    return alternative[0],other


def choose_terminal(det,segments):
    raw=det.get('raw_bbox_xyxy',det['body_bbox_xyxy']);cx,cy=center(raw)
    boundary=[(cx,raw[1]),(cx,raw[3]),(raw[0],cy),(raw[2],cy)]
    options=[]
    for index,s in enumerate(segments):
        if s['end']-s['start']<20:continue
        for endpoint in (s['start'],s['end']):
            point=[s['axis'],endpoint] if s['direction']=='vertical' else [endpoint,s['axis']]
            if s['direction']=='vertical':
                if abs(point[0]-cx)>max(12,(raw[2]-raw[0])*.35):continue
            elif abs(point[1]-cy)>max(12,(raw[3]-raw[1])*.35):continue
            distance=min(math.dist(point,p) for p in boundary)
            if distance<=18:options.append((distance-.02*min(100,s['end']-s['start']),index,point))
    return min(options,key=lambda p:p[0]) if options else None


def choose_missing_rail_label(det,items,ics):
    """Only an explicit high-confidence voltage-rail spelling missed by old rules."""
    if det['class_name']!='v' or det.get('confidence',0)<.5:return None
    raw=det.get('raw_bbox_xyxy',det['body_bbox_xyxy'])
    if inside_ic(center(raw),ics):return None
    options=[]
    for item in items:
        if item.get('confidence',0)<.95 or inside_ic(center(item['bbox_xyxy']),ics):continue
        if box_distance(raw,item['bbox_xyxy'])>20:continue
        if not re.fullmatch(r'(?:VBUS|V(?:CC|DD|SS)_?\d+\.\d+V)',item['text'].strip(),re.I):continue
        options.append((math.dist(center(raw),center(item['bbox_xyxy'])),item))
    options.sort(key=lambda p:p[0])
    if not options:return None
    if any(item['text'].strip().upper()!=options[0][1]['text'].strip().upper()
           and distance-options[0][0]<=5 for distance,item in options[1:]):return None
    return options[0][1],'v'


def choose_short_rail_terminal(det,label,segments,mask,roots):
    """Observe a short vertical rail stem, its transverse bar and a separate wire.

    Restricted to the explicit missing-label fallback. Never reduce the ordinary
    terminal length threshold or attach a rail from text alone.
    """
    if choose_missing_rail_label(det,[label],[]) is None:return None
    raw=det.get('raw_bbox_xyxy',det['body_bbox_xyxy']);cx,cy=center(raw);options={}
    def segment_distance(point,s):
        if s['direction']=='vertical':return math.hypot(point[0]-s['axis'],max(s['start']-point[1],point[1]-s['end'],0))
        return math.hypot(point[1]-s['axis'],max(s['start']-point[0],point[0]-s['end'],0))
    for index,stem in enumerate(segments):
        length=stem['end']-stem['start']
        if stem['direction']!='vertical' or not 8<=length<20 or abs(stem['axis']-cx)>3:continue
        for near,far in [(stem['start'],stem['end']),(stem['end'],stem['start'])]:
            point=[stem['axis'],near];other=[stem['axis'],far]
            if abs(near-cy)>8 or abs(far-cy)<abs(near-cy)+7:continue
            x=round(stem['axis']);ys=range(round(stem['start']),round(stem['end'])+1)
            if not 1<=x<mask.shape[1]-1 or any(y<0 or y>=mask.shape[0] for y in ys):continue
            if sum(bool(mask[y,x-1:x+2].any()) for y in ys)/len(ys)<.85:continue
            for bar_index,bar in enumerate(segments):
                if bar['direction']!='horizontal' or not 6<=bar['end']-bar['start']<=30:continue
                if abs(bar['axis']-near)>2 or not bar['start']+2<=stem['axis']<=bar['end']-2:continue
                if abs((bar['start']+bar['end'])/2-stem['axis'])>3:continue
                supports=[s['id'] for j,s in enumerate(segments) if j not in (index,bar_index)
                          and roots[j]==roots[index]==roots[bar_index]
                          and s['end']-s['start']>=20 and segment_distance(other,s)<=8]
                if not supports:continue
                distance=min(math.dist(point,p) for p in [(cx,raw[1]),(cx,raw[3]),(raw[0],cy),(raw[2],cy)])
                if distance<=8:options[(index,near)]=(distance-.02*length,index,point)
    return next(iter(options.values())) if len(options)==1 else None


def choose_local_label(det,items,ics):
    raw=det.get('raw_bbox_xyxy',det['body_bbox_xyxy']);options=[]
    for item in items:
        if item['confidence']<.9 or inside_ic(center(item['bbox_xyxy']),ics):continue
        if box_distance(item['bbox_xyxy'],raw)>20:continue
        kind='gnd' if valid_label(item['text'],'gnd') else 'v' if valid_label(item['text'],'v') else None
        if kind:options.append((math.dist(center(raw),center(item['bbox_xyxy'])),item,kind))
    options.sort(key=lambda item:item[0])
    if not options:return None
    best=options[0]
    if any(item[1]['text']!=best[1]['text'] and item[0]-best[0]<=5 for item in options[1:]):return None
    return best[1],best[2]


def duplicate_symbol(det,kind,point,attached):
    cx,cy=center(det['body_bbox_xyxy'])
    for symbol in attached:
        if symbol['type']!=kind:continue
        if math.dist([cx,cy],center(symbol['body_bbox_xyxy']))<=12 and math.dist(point,symbol['contact_xy'])<=8:return True
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('networks','pins','detections','ocr','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    original=json.loads(a.networks.read_text(encoding='utf-8'));result=copy.deepcopy(original)
    pins=json.loads(a.pins.read_text(encoding='utf-8'))
    detections=json.loads(a.detections.read_text(encoding='utf-8'))['versions']['filtered']
    items=json.loads(a.ocr.read_text(encoding='utf-8'))['items']
    image=Image.open(original['image']).convert('RGB');mask=np.asarray(image.convert('L'))<160
    if original.get('bright_wire_runs'):
        from bright_wire_pixels import restore_runs
        restore_runs(mask,original['bright_wire_runs'])
    segments=result['segments'];uf,_=connect_segments(segments,mask);id_to_index={s['id']:i for i,s in enumerate(segments)}
    group_for_root={}
    for n in result['networks']:
        if n['segment_ids']:group_for_root[uf.find(id_to_index[n['segment_ids'][0]])]=n
    proposals=[];unresolved=[];local_ocr=[];engine=None;local_errors=[];local_attempts=0;rail_recovery={};ground_mask=None
    for d in detections:
        if d['class_name'] not in ('gnd','v'):continue
        if inside_ic(center(d['body_bbox_xyxy']),pins['ics']):
            unresolved.append(dict(id=d['id'],reason='low_confidence_or_inside_ic'));continue
        if d['confidence']<.5:
            from low_confidence_power import recover_low_voltage
            from weak_ground import recover_weak_ground,ground_ink_mask
            if d['class_name']=='gnd' and ground_mask is None:ground_mask=ground_ink_mask(image)
            recovered=(recover_weak_ground(d,items,pins['ics'],segments,ground_mask) if d['class_name']=='gnd'
                       else recover_low_voltage(d,items,pins['ics'],segments,mask,uf,group_for_root))
            if recovered is None:
                unresolved.append(dict(id=d['id'],reason='low_confidence_without_independent_voltage_evidence'));continue
            selected_label,terminal,evidence=recovered
            rail_recovery[d['id']]=evidence
            score,index,point=terminal
            proposals.append((score-3*d['confidence'],d,selected_label[0],index,point,selected_label[1]))
            continue
        selected_label=choose_label(d,items,pins['ics'])
        terminal=choose_terminal(d,segments)
        if selected_label is None:
            selected_label=choose_missing_rail_label(d,items,pins['ics'])
            if selected_label is not None:
                short=False
                if terminal is None:
                    terminal=choose_short_rail_terminal(d,selected_label[0],segments,mask,[uf.find(i) for i in range(len(segments))]);short=terminal is not None
                rail_recovery[d['id']]=dict(source='explicit_missing_voltage_rail_label',
                    observed_short_stem_bar_and_wire=short,ocr_bbox_xyxy=selected_label[0]['bbox_xyxy'])
        if selected_label is None and terminal is not None and local_attempts<24:
            local_attempts+=1
            try:
                if engine is None:
                    from rapidocr_onnxruntime import RapidOCR
                    engine=RapidOCR()
                from power_label_ocr import local_labels
                recovered=local_labels(engine,image,d,pins['ics']);local_ocr.extend(recovered)
                selected_label=choose_local_label(d,recovered,pins['ics'])
            except Exception as exc:
                local_errors.append(dict(detection_id=d['id'],error=type(exc).__name__,message=str(exc)))
        if selected_label is None or terminal is None:
            unresolved.append(dict(id=d['id'],reason='no_supported_label_or_wire_endpoint'));continue
        label,kind=selected_label
        score,index,point=terminal
        proposals.append((score-3*d['confidence'],d,label,index,point,kind))
    used_text=set();attached=[]
    for _,d,label,index,point,kind in sorted(proposals,key=lambda p:(p[1]['confidence']<.5,p[2].get('source')=='local_rotated_ocr',p[0])):
        if label['id'] in used_text:
            unresolved.append(dict(id=d['id'],reason='same_ocr_label_already_claimed'));continue
        if (d['confidence']<.5 or label.get('source')=='local_rotated_ocr') and duplicate_symbol(d,kind,point,attached):
            unresolved.append(dict(id=d['id'],reason='same_geometric_symbol_already_attached'));continue
        used_text.add(label['id']);root=uf.find(index)
        if root not in group_for_root:
            ids=[s['id'] for i,s in enumerate(segments) if uf.find(i)==root]
            n=dict(id=f'power_only_group_{len(group_for_root):03d}',pins=[],segment_ids=ids)
            result['networks'].append(n);group_for_root[root]=n
        ref=dict(kind='power',component_id=d['id'],type=kind,name=label['text'],number='terminal_1',
                 contact_xy=point,segment_id=segments[index]['id'],body_bbox_xyxy=d['body_bbox_xyxy'],
                 detector_confidence=d['confidence'],label_ocr_id=label['id'],detector_original_type=d['class_name'],
                 category_source='detector_and_label' if kind==d['class_name'] else 'nearby_explicit_ocr_overrides_detector',
                 label_confidence=label['confidence'],label_source=label.get('source','full_image_ocr'))
        if d['id'] in rail_recovery:ref['rail_recovery_evidence']=rail_recovery[d['id']]
        group_for_root[root]['pins'].append(ref);attached.append(dict(**ref,wire_group_id=group_for_root[root]['id']))
    # Detector-independent recovery still requires OCR, bar geometry and a wire.
    from ground_shape149 import proposals as observed_ground_proposals
    ground_recovery=[]
    for shape in observed_ground_proposals(image,items):
        if len(ground_recovery)>=12:break
        b=shape['bbox'];cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
        if inside_ic([cx,cy],pins['ics']):continue
        if any(math.dist([cx,cy],center(s['body_bbox_xyxy']))<30 for s in attached if s['type']=='gnd'):continue
        label=next((t for t in items if t['id']==shape['ocr_id']),None)
        if label is None:continue
        d=dict(class_name='gnd',confidence=min(.99,label['confidence']),body_bbox_xyxy=b,raw_bbox_xyxy=b)
        from ground_terminal151 import choose_ground_terminal
        terminal=choose_ground_terminal(shape,segments,mask)
        if terminal is None:continue
        score,index,point=terminal;root=uf.find(index)
        if root not in group_for_root:
            ids=[s['id'] for i,s in enumerate(segments) if uf.find(i)==root]
            n=dict(id=f'power_only_group_{len(group_for_root):03d}',pins=[],segment_ids=ids)
            result['networks'].append(n);group_for_root[root]=n
        ref=dict(kind='power',component_id='ground_shape_'+str(len(ground_recovery)),type='gnd',name=label['text'],number='terminal_1',contact_xy=point,segment_id=segments[index]['id'],body_bbox_xyxy=b,detector_confidence=d['confidence'],label_ocr_id=label['id'],detector_original_type='gnd',category_source='observed_three_ground_bars_and_OCR',label_confidence=label['confidence'],label_source='full_image_ocr',ground_shape_evidence=shape)
        group_for_root[root]['pins'].append(ref);attached.append(dict(**ref,wire_group_id=group_for_root[root]['id']));ground_recovery.append(ref)
    result['ground_shape_recovery']=dict(added=ground_recovery,max_added=12)
    # Identical text is metadata, not permission to merge physically distinct groups.
    named={}
    for symbol in attached:named.setdefault(symbol['name'],set()).add(symbol['wire_group_id'])
    result['power_symbols']=attached;result['unresolved_power_candidates']=unresolved
    result['same_label_distinct_group_candidates']={k:sorted(v) for k,v in named.items() if len(v)>1}
    result['summary'].update(attached_power_symbols=len(attached),unresolved_power_candidates=len(unresolved),
                             wire_groups_with_observed_contacts=len(result['networks']))
    result['power_label_policy']='Attach text to physical groups; do not merge same-name labels without verified annotation/scoring convention.'
    result['limitations']+=' Power contacts are observed line endpoints near detected symbols; not yet verified official symbol pin coordinates.'
    (a.output/'wire_networks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'local_label_ocr.json').write_text(json.dumps(dict(items=local_ocr,attempted_candidates=local_attempts,
        candidate_limit=24,errors=local_errors),ensure_ascii=False,indent=2),encoding='utf-8')
    draw=ImageDraw.Draw(image)
    for s in attached:
        x,y=s['contact_xy'];draw.ellipse((x-4,y-4,x+4,y+4),outline='red',width=2)
        draw.text((x+5,y),s['name'],fill='blue')
    image.save(a.output/'power_contacts.png');print(json.dumps(dict(attached=len(attached),unresolved=len(unresolved))))


if __name__=='__main__':main()
