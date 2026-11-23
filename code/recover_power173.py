"""Bounded image-only voltage recovery after the main CV candidate pipeline."""
import argparse,json,time,hashlib
from pathlib import Path
from PIL import Image
from power_evidence173 import proposals,gate
from power_label_ocr import local_labels
from attach_power_symbols import valid_label

def recover(candidate,image,engine):
    c=candidate;events=[];attempts=0
    excluded=[x['body_bbox_xyxy'] for x in c['passives']]+[x['body_bbox_xyxy'] for x in c['ics']]+[x['body_bbox_xyxy'] for x in c['networks']['power_symbols']]
    for i,p in enumerate(proposals(image)):
        b=p['body_bbox_xyxy']
        if any(max(b[0]-3,e[0])<min(b[2]+3,e[2]) and max(b[1]-3,e[1])<min(b[3]+3,e[3]) for e in excluded):continue
        if attempts>=6:break
        attempts+=1;d=dict(p,id='shape_'+str(i),class_name='v')
        labels=local_labels(engine,image,d,c['ics'])
        record=dict(proposal=p,high_confidence_voltage=[t for t in labels if valid_label(t['text'],'v') and t['confidence']>=.95])
        evidence,reason=gate(record,c);events.append(dict(**record,reason=reason))
        if evidence is None:continue
        label,seg,group=evidence;cid='observed_voltage_'+str(i)
        if group.get('isolated_voltage_group') and not any(n['id']==group['id'] for n in c['networks']['networks']):
            c['networks']['networks'].append(group)
        ref=dict(kind='power',component_id=cid,type='v',name=label['text'],number='terminal_1',contact_xy=p['wire_start'],segment_id=seg['id'],body_bbox_xyxy=b,pin_tip_candidate=None,
                 category_source='observed_bar_stem_wire_and_two_rotation_voltage_text',recovery_evidence=record)
        group['pins'].append(ref);c['networks']['power_symbols'].append(ref)
    c['networks']['voltage_recovery173']=dict(events=events,attempts=attempts,max_attempts=6)
    return events,attempts

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pipeline',type=Path,required=True);args=parser.parse_args()
    from rapidocr_onnxruntime import RapidOCR
    engine=RapidOCR();rows=[];start=time.time()
    for path in sorted((args.pipeline/'candidates').glob('*.json')):
        c=json.loads(path.read_text(encoding='utf-8'));image_path=Path(c['input']['image'])
        assert hashlib.sha256(image_path.read_bytes()).hexdigest()==c['input']['image_sha256']
        with Image.open(image_path) as im:events,attempts=recover(c,im.convert('RGB'),engine)
        path.write_text(json.dumps(c,indent=2,ensure_ascii=False),encoding='utf-8')
        rows.append(dict(case=path.stem,attempts=attempts,added=sum(e['reason'] in ('accepted','accepted_isolated_symbol') for e in events)))
    result=dict(cases=rows,seconds=time.time()-start,API_calls=0,max_OCR_calls_per_image=24)
    (args.pipeline/'voltage_recovery173.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    p=args.pipeline/'decision_log.json';d=json.loads(p.read_text(encoding='utf-8'));d['voltage_recovery173']=result;p.write_text(json.dumps(d,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
