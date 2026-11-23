"""Check task identity, stale inputs and structured proposals; never apply them."""
import hashlib
import json
import math
import re
from pathlib import Path
from PIL import Image


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_reply(task,reply):
    if not isinstance(reply,dict) or set(reply)!= {'task_id','candidate_sha256','decision','reason','proposal'}:
        raise ValueError('Reply schema')
    if reply['task_id']!=task['task_id'] or reply['candidate_sha256']!=task['candidate_sha256']:
        raise ValueError('Reply/task identity mismatch')
    candidate_path=Path(task['candidate_path'])
    if sha(candidate_path)!=task['candidate_sha256']:raise ValueError('Stale candidate')
    candidate=json.loads(candidate_path.read_text(encoding='utf-8'))
    if candidate['case_name']!=task['case_name']:raise ValueError('Case identity mismatch')
    image_path=Path(candidate['input']['image'])
    if sha(image_path)!=task['image_sha256'] or sha(task['crop_path'])!=task['crop_sha256']:
        raise ValueError('Stale image or crop')
    for asset in candidate['provenance'].values():
        if sha(asset['path'])!=asset['sha256']:raise ValueError('Stale model/library/prior asset')
    library=candidate['provenance'].get('library')
    if library and library['sha256']!=task['library_sha256']:raise ValueError('Library/task mismatch')
    decision=reply['decision']
    if decision not in ('keep','correct','uncertain','reject'):raise ValueError('Unknown decision')
    if not isinstance(reply['reason'],str) or not 1<=len(reply['reason'])<=1500:raise ValueError('Reason')
    proposal=reply['proposal']
    if decision!='correct':
        if proposal is not None:raise ValueError('Only correct may contain a proposal')
    else:
        allowed={'designator','model','value','pin_text'} if task.get('review_focus')=='text' else {'designator','model','value','pin_positions'}
        if not isinstance(proposal,dict) or not proposal or not set(proposal)<=allowed:
            raise ValueError('Proposal fields')
        for field in ('designator','model','value'):
            if field in proposal and (not isinstance(proposal[field],str) or not 1<=len(proposal[field])<=120):
                raise ValueError('Proposed text')
        if 'pin_text' in proposal:
            edits=proposal['pin_text'];context=task['context']
            if task['target_kind']!='ic' or not isinstance(edits,list) or not 1<=len(edits)<=2:raise ValueError('Pin text schema')
            known={pin['number']:pin for pin in context['pins']};used=set();updated={number:number for number in known}
            crop=task['crop_xyxy']
            for edit in edits:
                if not isinstance(edit,dict) or set(edit)!={'source_number','number','pinname','evidence_bbox_xyxy'}:raise ValueError('Pin text schema')
                source=edit['source_number']
                if not isinstance(source,str) or source not in known or source in used:raise ValueError('Pin text source row')
                used.add(source);number=edit['number'];name=edit['pinname']
                if not isinstance(number,str) or re.fullmatch(r'[0-9]{1,5}',number) is None or not isinstance(name,str) or not 1<=len(name)<=120:
                    raise ValueError('Pin text number/name')
                box=edit['evidence_bbox_xyxy']
                if not isinstance(box,list) or len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box):
                    raise ValueError('Pin text evidence box')
                if not(crop[0]<=box[0]<box[2]<=crop[2] and crop[1]<=box[1]<box[3]<=crop[3]):raise ValueError('Pin text evidence box')
                pin=known[source];contact=pin.get('body_contact_xy');side=pin.get('side')
                if not contact or side not in ('left','right','top','bottom'):raise ValueError('Pin text evidence outside row')
                axis=1 if side in ('left','right') else 0
                if box[axis+2]-box[axis]>80 or not(box[axis]-20<=contact[axis]<=box[axis+2]+20):raise ValueError('Pin text evidence outside row')
                updated[source]=number
            if len(set(updated.values()))!=len(updated):raise ValueError('Pin text number collision')
        if 'pin_positions' in proposal:
            pins=proposal['pin_positions'];context=task['context'];kind=task['target_kind']
            if kind=='ic':known={pin['number'] for pin in context['pins']}
            elif kind=='passive':known={'1','2'}
            elif kind=='power':known={'1'}
            else:known=set()
            if not isinstance(pins,list) or not 1<=len(pins)<=100:raise ValueError('Pin proposals')
            used=set();crop=task['crop_xyxy']
            with Image.open(image_path) as image:width,height=image.size
            for pin in pins:
                if not isinstance(pin,dict) or set(pin)!= {'number','position_image_xy'}:raise ValueError('Pin proposal schema')
                number=pin['number']
                if not isinstance(number,str) or number not in known or number in used:raise ValueError('Unknown/duplicate proposed pin')
                used.add(number);xy=pin['position_image_xy']
                if not isinstance(xy,list) or len(xy)!=2:raise ValueError('Proposed point')
                if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in xy):
                    raise ValueError('Proposed coordinate number')
                if not (0<=xy[0]<=width and 0<=xy[1]<=height and crop[0]<=xy[0]<=crop[2] and crop[1]<=xy[1]<=crop[3]):
                    raise ValueError('Proposed point outside image/crop context')
    return dict(task_id=task['task_id'],decision=decision,status='proposal_not_applied' if decision=='correct' else 'review_recorded_not_ground_truth',
                applied=False,proposal=proposal,reason=reply['reason'])
