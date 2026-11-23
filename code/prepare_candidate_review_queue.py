"""Build case-isolated local review tasks with hashes and bounded selection; no API calls."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from PIL import Image


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ic_text_reasons(ic,library):
    reasons=[]
    if not ic.get('model'):reasons.append('missing_model_text')
    elif ic['model'] not in library:reasons.append('model_text_not_in_primary_library')
    if not ic.get('designator'):reasons.append('missing_designator_text')
    if not ic.get('pins'):reasons.append('no_observed_pin_text_pairs')
    if any(min(pin.get('number_confidence',1),pin.get('name_confidence',1))<.85 for pin in ic.get('pins',[])):
        reasons.append('low_confidence_observed_pin_text')
    if any(pin.get('library_status') in ('name_conflict','number_missing','name_mismatch','conflict','number_not_in_library') for pin in ic.get('pins',[])):
        reasons.append('observed_pin_text_library_conflict')
    if ic.get('unused_number_ocr_ids') and ic.get('unused_name_ocr_ids'):
        reasons.append('unpaired_number_and_name_ocr')
    return reasons


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-selected-per-case',type=int,default=3)
    p.add_argument('--focus',choices=('text','geometry'),default='text')
    a=p.parse_args()
    if not 0<=a.max_selected_per_case<=5:p.error('Selection limit must be 0..5')
    if a.output.exists() and any(a.output.iterdir()):p.error('Use an empty output directory')
    a.output.mkdir(parents=True,exist_ok=True);cases=[]
    for file in sorted(a.candidates.glob('*.json')):
        c=json.loads(file.read_text(encoding='utf-8'));case=c['case_name'];image_path=Path(c['input']['image'])
        if file.stem!=case or sha(image_path)!=c['input']['image_sha256']:raise ValueError('Case identity or image hash')
        asset=c['provenance']['library'];library_path=Path(asset['path'])
        if sha(library_path)!=asset['sha256']:raise ValueError('Library changed')
        library=json.loads(library_path.read_text(encoding='utf-8-sig'))
        image=Image.open(image_path).convert('RGB');folder=a.output/case;folder.mkdir();tasks=[]
        candidate_hash=sha(file)
        def add(kind,target,box,priority,reasons,context):
            region=[max(0,math.floor(box[0]-100)),max(0,math.floor(box[1]-100)),
                    min(image.width,math.ceil(box[2]+101)),min(image.height,math.ceil(box[3]+101))]
            task_id=hashlib.sha256(json.dumps([case,candidate_hash,kind,target,a.focus],ensure_ascii=False).encode()).hexdigest()[:20]
            crop_path=folder/(task_id+'.png');image.crop(region).save(crop_path)
            task=dict(task_id=task_id,case_name=case,target_kind=kind,target_id=target,priority=priority,reasons=reasons,
                      candidate_sha256=candidate_hash,candidate_path=str(file.resolve()),image_sha256=c['input']['image_sha256'],
                      library_sha256=asset['sha256'],crop_sha256=sha(crop_path),crop_path=str(crop_path.resolve()),crop_xyxy=region,
                      coordinate_system='original full image pixels; top-left origin; never return crop coordinates',
                      context=context,review_focus=a.focus,status='pending_local_only',selected=False,
                      instructions='Inspect visible evidence only. Continuous black pin/wire lines may have no visible boundary. Return uncertain when geometry cannot identify it. Do not invent missing pins from library, merge same-name physical groups, or use task priority as confidence. Corrections are proposals; no automatic application.',
                      response_fields=['decision','reason','proposal'],
                      decisions=['keep','correct','uncertain','reject'],max_output_tokens=768)
            if a.focus=='text':
                task['instructions']='Check only visible text. Library absence is not proof of OCR error. Existing row text edits need local evidence boxes. No new rows, tip coordinates or wire changes. Proposals are not applied.'
            tasks.append(task)
        for ic in c['ics']:
            if a.focus=='text':
                reasons=ic_text_reasons(ic,library)
                if reasons:add('ic',ic['detection_id'],ic['body_bbox_xyxy'],90 if ic['model'] not in library else 80,reasons,ic)
                continue
            reasons=[]
            if ic['model'] not in library:reasons.append('model_missing_or_not_in_primary_library')
            if not ic['designator']:reasons.append('missing_designator')
            if not ic['pins']:reasons.append('no_pin_text_pairs')
            if any(pin.get('tip_geometry',{}).get('position_image_xy') is None for pin in ic['pins']):
                reasons.append('ic_pin_wire_boundary_unresolved')
            if reasons:add('ic',ic['detection_id'],ic['body_bbox_xyxy'],90 if ic['model'] not in library else 80,reasons,ic)
        for component in c['passives']:
            if a.focus=='text':
                reasons=[name for field,name in [('designator','missing_designator_text'),('value','missing_value_text')] if not component.get(field)]
                if reasons:add('passive',component['id'],component['body_bbox_xyxy'],70,reasons,component)
                continue
            if component['accepted'] and component.get('pin_tip_status')=='two_image_supported_offset_hypotheses':continue
            add('passive',component['id'],component['body_bbox_xyxy'],70,
                [component.get('pin_tip_status','unresolved_passive')],component)
        for symbol in c['networks']['power_symbols']:
            if a.focus=='text':
                if not symbol.get('name'):add('power',symbol['component_id'],symbol['body_bbox_xyxy'],60,['missing_power_label_text'],symbol)
                continue
            if symbol.get('pin_tip_candidate') is not None:continue
            add('power',symbol['component_id'],symbol['body_bbox_xyxy'],60,[symbol.get('pin_tip_status','unresolved_power')],symbol)
        for detection in c['unsupported_detections']:
            if a.focus=='text':continue
            if detection['class_name'] in ('v','gnd'):continue
            add('unsupported_component',detection['id'],detection['body_bbox_xyxy'],50,
                ['component_class_not_parsed'],detection)
        tasks.sort(key=lambda task:(-task['priority'],task['task_id']))
        for task in tasks[:a.max_selected_per_case]:task['selected']=True
        report=dict(case_name=case,candidate_sha256=candidate_hash,review_focus=a.focus,tasks=tasks,
                    selected_task_ids=[task['task_id'] for task in tasks if task['selected']],
                    kind_counts=dict(Counter(task['target_kind'] for task in tasks)),
                    api_calls=0,note='Local plan only. Selected does not mean sent or approved for image upload. New task schema requires a matching client/reply validator; do not feed directly into the old component-text smoke test.')
        (folder/'review_queue.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        cases.append(dict(case_name=case,tasks=len(tasks),selected=len(report['selected_task_ids']),kind_counts=report['kind_counts']))
    manifest=dict(cases=cases,review_focus=a.focus,api_calls=0,max_selected_per_case=a.max_selected_per_case,
                  max_selected_output_tokens=sum(case['selected'] for case in cases)*768,
                  note='Selection and output-token cap only, not a price or input/image token estimate. No network requests.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
