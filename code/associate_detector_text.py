"""Join detector outputs with image OCR; no GT argument or GT reads."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
from text_association import associate, PREFIX, box_distance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('image','detections','ocr','library','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--variant',default='tiled_prior_merged')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    image=Image.open(a.image).convert('RGB')
    raw=json.loads(a.detections.read_text(encoding='utf-8'))
    detections=raw['versions'][a.variant] if 'versions' in raw else raw['detections']
    ocr=json.loads(a.ocr.read_text(encoding='utf-8'))
    if ocr.get('crop') != [0,0,image.width,image.height]:
        raise ValueError('Expected cached full-image OCR in original image coordinates')
    library=json.loads(a.library.read_text(encoding='utf-8-sig'))
    supported=[d for d in detections if d['class_name'] in PREFIX]
    if len({d['id'] for d in detections})!=len(detections):raise ValueError('Duplicate detection IDs')
    from wire_text_ownership198 import mark
    ocr['items']=mark(ocr['items'],a.image)
    predictions,typed,used=associate(supported,ocr['items'],library,100,20)
    by_id={d['id']:d for d in detections};tasks=[]
    def review(kind,identifier,box,reasons,payload):
        left,top,right,bottom=box
        crop=[max(0,int(left)-80),max(0,int(top)-80),min(image.width,int(right)+81),min(image.height,int(bottom)+81)]
        folder=a.output/'review_crops';folder.mkdir(exist_ok=True)
        dest=folder/(f'{len(tasks):03d}_{identifier}.png');image.crop(crop).save(dest)
        tasks.append(dict(task_id=f'review_{len(tasks):04d}',kind=kind,target_id=identifier,reasons=reasons,
                          crop_path=str(dest.resolve()),crop_xyxy=crop,payload=payload,status='pending_no_api_call'))
    for pred in predictions:
        det=by_id[pred['detection_id']]
        pred.update(body_bbox_xyxy=det['body_bbox_xyxy'],detector_confidence=det['confidence'],source='detector_plus_ocr')
        reasons=list(det.get('candidate_review_reasons',[]))
        field='model' if pred['type']=='box' else 'value'
        for key in ('designator',field):
            if pred[key] is None:reasons.append('missing_'+key)
        if det['confidence']<.5:reasons.append('low_detector_confidence')
        if pred.get('library_status')=='unresolved':reasons.append('model_not_in_library')
        if any(e['ocr_confidence']<.85 for e in pred['evidence'].values()):reasons.append('low_ocr_confidence')
        # One-to-one greedy matching can assign a label to the wrong nearby proposal.
        if any(d['id']!=det['id'] and d['class_name']==det['class_name'] and
               box_distance(d['body_bbox_xyxy'],det['body_bbox_xyxy'])<20 for d in supported):
            reasons.append('nearby_same_class_candidate')
        pred['review_reasons']=reasons
        if reasons:
            near=[t for t in typed if box_distance(t['bbox_xyxy'],det['body_bbox_xyxy'])<=100]
            review('component',det['id'],det['body_bbox_xyxy'],reasons,dict(prediction=pred,nearby_ocr=near))
    for item in typed:
        if item['role_candidate']=='designator' and item['id'] not in used:
            review('unassigned_designator',item['id'],item['bbox_xyxy'],['possible_missed_component_or_wrong_association'],dict(ocr=item))
    report=dict(image=str(a.image.resolve()),variant=a.variant,predictions=predictions,
                unsupported_detections=[d for d in detections if d['class_name'] not in PREFIX],
                summary=dict(all_detections=len(detections),supported_components=len(predictions),
                             complete_text_components=sum(p['designator'] is not None and p['model' if p['type']=='box' else 'value'] is not None for p in predictions),
                             review_tasks=len(tasks)),
                provenance=dict(detector_file=str(a.detections.resolve()),ocr_file=str(a.ocr.resolve()),
                                detector_sha256=hashlib.sha256(a.detections.read_bytes()).hexdigest(),
                                library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest()),
                limitations='r/c/box with image-observed R/C and IC/connector designators; cached image OCR; heuristic association. A filled field is not confirmed correct. No GT or API used.')
    (a.output/'associations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'review_queue.json').write_text(json.dumps(dict(tasks=tasks,note='Local candidates only. Do not send the whole queue automatically; select with a token budget and validate responses.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary']))


if __name__=='__main__':main()
