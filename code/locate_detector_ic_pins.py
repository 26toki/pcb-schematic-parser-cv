"""Observe IC rectangles and horizontal pin contacts from image pixels; no GT inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from text_association import associate
from ic_text_pairing import pair_text


def runs(column):
    padded=np.r_[False,column,False].astype(np.int8)
    edges=np.diff(padded)
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def border_column(mask,x,y1,y2,radius=30):
    height,width=mask.shape;span=max(1,y2-y1);proposals=[]
    for xx in range(max(0,round(x)-radius),min(width,round(x)+radius+1)):
        for start,end in runs(mask[:,xx]):
            overlap=max(0,min(end,y2)-max(start,y1))
            if overlap/span>=.6 and end-start>=max(25,.6*span):
                score=overlap/span-.01*abs(xx-x)+.0001*(end-start)
                proposals.append((score,xx,int(start),int(end)))
    return max(proposals,default=None)


def rectangle_from_pixels(mask,box):
    x1,y1,x2,y2=box
    left=border_column(mask,x1,y1,y2);right=border_column(mask,x2,y1,y2)
    if left is None or right is None:return None
    _,lx,lt,lb=left;_,rx,rt,rb=right
    if rx-lx<25 or abs(lt-rt)>6 or abs(lb-rb)>6:return None
    top=int(round((lt+rt)/2));bottom=int(round((lb+rb)/2))-1
    if bottom-top<25:return None
    for yy in (top,bottom):
        if mask[max(0,yy-1):min(mask.shape[0],yy+2),lx:rx+1].any(axis=0).mean()<.85:return None
    return [lx,top,rx,bottom]


def observe_contact(mask,box,pair,texts):
    side=pair['side'];edge=box[0] if side=='left' else box[2]
    label=texts[pair['name_ocr_id']]['bbox_xyxy'];target=(label[1]+label[3])/2
    height,width=mask.shape;sign=-1 if side=='left' else 1;proposals=[]
    for y in range(max(box[1]+3,round(target)-12),min(box[3]-2,round(target)+13)):
        xs=[edge+sign*k for k in range(3,10)]
        if any(x<0 or x>=width for x in xs):continue
        coverage=float(mask[y,xs].mean())
        if coverage<.85:continue
        proposals.append((abs(y-target)-.2*coverage,y))
    result={**pair,'body_contact_xy':None,'pin_position':None,'position_status':'no_observed_horizontal_contact'}
    if not proposals:return result
    _,y=min(proposals)
    result['body_contact_xy']=[edge,y]
    result['position_status']='body_contact_observed_tip_unresolved'
    # Trace along the observed row. Long wires/junctions cannot identify symbol tips.
    x=edge+sign*3;last=x;length=0
    while 0<=x<width and mask[max(0,y-1):min(height,y+2),x].any() and length<=80:
        last=x;x+=sign;length+=1
    result['observed_outside_run_length_px']=length
    if length<=60 and length>=5 and 0<x<width-1:
        # A vertical branch at the end is a wire junction, not a free endpoint.
        vertical=np.r_[mask[max(0,y-6):max(0,y-2),max(0,last-1):min(width,last+2)].ravel(),
                       mask[min(height,y+3):min(height,y+7),max(0,last-1):min(width,last+2)].ravel()]
        if not vertical.any():
            result['pin_position']=[last,y]
            result['position_status']='observed_free_line_end_candidate'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('image','associations','ocr','library','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--local-ocr',action='store_true',help='OCR crops derived solely from observed image rectangles')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    image=Image.open(a.image).convert('RGB');mask=np.asarray(image.convert('L'))<160
    associations=json.loads(a.associations.read_text(encoding='utf-8'))['predictions']
    items=json.loads(a.ocr.read_text(encoding='utf-8'))['items']
    library=json.loads(a.library.read_text(encoding='utf-8-sig'))
    verified=[];unverified=[]
    for pred in associations:
        if pred['type']!='box':continue
        observed=rectangle_from_pixels(mask,pred['body_bbox_xyxy'])
        if observed is None:unverified.append(pred['detection_id']);continue
        found=next((d for d in verified if max(abs(x-y) for x,y in zip(d['body_bbox_xyxy'],observed))<=4),None)
        if found:found['source_detection_ids'].append(pred['detection_id'])
        else:verified.append(dict(id=f'ic_{len(verified)+1:04d}',class_name='box',body_bbox_xyxy=observed,
                                  confidence=pred['detector_confidence'],source_detection_ids=[pred['detection_id']]))
    from wire_text_ownership198 import mark
    items=mark(items,a.image)
    text_predictions,_,_=associate(verified,items,library,100,20)
    by_id={p['detection_id']:p for p in text_predictions};texts={t['id']:t for t in items}
    engine=None
    if a.local_ocr:
        from rapidocr_onnxruntime import RapidOCR
        from read_schematic_text import recognize
        engine=RapidOCR()
    output=[];canvas=image.copy();draw=ImageDraw.Draw(canvas);reference_attempts=0;reference_events=[]
    for d in verified:
        pred=by_id[d['id']];box=d['body_bbox_xyxy']
        pairs,unused_numbers,unused_names=pair_text(box,items,library.get(pred['model'],{}),100,20)
        for pair in pairs:pair['text_source']='full_image_ocr'
        local_candidates=[];reference_event=None
        if engine is not None:
            region=[max(0,box[0]-50),max(0,box[1]-50),min(image.width,box[2]+50),min(image.height,box[3]+50)]
            folder=a.output/'local_ocr'/d['id']
            recognize(engine,image,region,3,folder)
            local_items=json.loads((folder/'ocr.json').read_text(encoding='utf-8'))['items']
            for item in local_items:item['id']=d['id']+'_local_'+item['id']
            local_items=mark(local_items,a.image)
            local_pred,_,_=associate([d],local_items,library,100,20)
            for field in ('designator','model'):
                # Do not replace a rejected wire label with another unverified
                # nearby text from a crop. Exact library models remain eligible.
                if (field == 'model' and pred.get('rejected_model_wire_evidence')
                        and local_pred[0][field] not in library):
                    continue
                if pred[field] is None and local_pred[0][field] is not None:pred[field]=local_pred[0][field]
            from reference_consensus157 import choose
            evidence=pred.get('evidence',{}).get('designator',{})
            old_reference=texts.get(evidence.get('ocr_id'))
            used={p.get('designator') for p in associations if p.get('designator')}
            used.update(p.get('designator') for p in text_predictions if p is not pred and p.get('designator'))
            used.update(p.get('designator') for p in output if p.get('designator'))
            if old_reference is not None and reference_attempts<4 and choose(old_reference,local_items,local_items,used):
                reference_attempts+=1;confirmation=folder/'reference_check4x'
                recognize(engine,image,region,4,confirmation)
                second=json.loads((confirmation/'ocr.json').read_text(encoding='utf-8'))['items']
                decision=choose(old_reference,local_items,second,used)
                if decision is not None:
                    pred['designator']=decision['after'];reference_event=dict(ic_id=d['id'],**decision);reference_events.append(reference_event)
            local_pairs,_,_=pair_text(box,local_items,library.get(pred['model'],{}),100,20)
            from numeric_connector185 import numeric_pairs
            numeric = numeric_pairs(box, local_items, pred.get('designator'), library.get(pred['model'], {}))
            numeric = [p for p in numeric if observe_contact(mask, box, p, {t['id']: t for t in local_items})['body_contact_xy'] is not None]
            if len(numeric) >= 3:
                from reconcile186 import reconcile
                pairs, numeric_replacements = reconcile(pairs, numeric)
                local_pairs.extend(numeric)
            number_counts=Counter(pair['number'] for pair in local_pairs)
            used_numbers={pair['number'] for pair in pairs}
            for pair in local_pairs:
                pair['text_source']='observed_rectangle_local_ocr'
                local_candidates.append(pair)
                if pair['number'] not in used_numbers and number_counts[pair['number']]==1 and (pair['library_status']=='exact_match' or pair.get('numeric_connector_evidence') is not None):
                    pairs.append(pair);used_numbers.add(pair['number'])
            texts.update({t['id']:t for t in local_items})
        pins=[observe_contact(mask,box,pair,texts) for pair in pairs]
        output.append(dict(detection_id=d['id'],source_detection_ids=d['source_detection_ids'],
                           designator=pred['designator'],model=pred['model'],body_bbox_xyxy=box,pins=pins,
                           local_pair_candidates_not_all_accepted=local_candidates,
                           unused_number_ocr_ids=unused_numbers,unused_name_ocr_ids=unused_names))
        if reference_event is not None:output[-1]['reference_recovery_evidence']=reference_event
        draw.rectangle(box,outline='blue',width=2)
        for pin in pins:
            if pin['body_contact_xy']:
                x,y=pin['body_contact_xy'];draw.ellipse((x-3,y-3,x+3,y+3),outline='red',width=2)
            if pin['pin_position']:
                x,y=pin['pin_position'];draw.ellipse((x-4,y-4,x+4,y+4),outline='green',width=2)
    summary=dict(verified_ics=len(output),unverified_box_candidates=len(unverified),
                 text_pairs=sum(len(ic['pins']) for ic in output),
                 observed_contacts=sum(pin['body_contact_xy'] is not None for ic in output for pin in ic['pins']),
                 free_endpoint_candidates=sum(pin['pin_position'] is not None for ic in output for pin in ic['pins']))
    report=dict(image=str(a.image.resolve()),summary=summary,ics=output,unverified_detection_ids=unverified,reference_recovery=dict(attempts=reference_attempts,max_attempts=4,events=reference_events),
                provenance=dict(association_sha256=hashlib.sha256(a.associations.read_bytes()).hexdigest()),
                local_ocr_enabled=a.local_ocr,
                limitations='Image-only prediction, no GT. Axis-aligned black rectangular ICs; left/right numeric pins. OCR omissions persist. Body contacts are not symbol tips. Free line ends are candidates, not guaranteed official pin coordinates. Thresholds fixed; not official scoring.')
    (a.output/'pins.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    canvas.save(a.output/'observed_pins.png');print(json.dumps(summary))


if __name__=='__main__':main()
