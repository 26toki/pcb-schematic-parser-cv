"""Pure image/text helpers. No training or GT entrypoint."""
import argparse
import json
import math
import re
from pathlib import Path
from PIL import Image, ImageDraw
PREFIX = {'r': 'R', 'c': 'C', 'l': 'L', 'box': ('U', 'USB', 'J', 'CN', 'P', 'H', 'FPC')}

def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def box_distance(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)

def classify(item, library):
    text = item['text'].strip()
    if re.fullmatch('(?:R|C|L|U|USB|J|CN|P|H|FPC)\\d+', text):
        return 'designator'
    if text in library:
        return 'model'
    if re.fullmatch('\\d+(?:\\.\\d+)?(?:-\\d+(?:\\.\\d+)?)?[pnuμµm]?[Ff]', text) or re.fullmatch('\\d+(?:\\.\\d+)?(?:[pnuμµm]?[Hh]|[kKMm]?[ΩΩ]|[kKM])', text):
        return 'value'
    if re.fullmatch('\\d+', text):
        return 'possible_pin_number'
    if len(text) >= 4 and re.search('[A-Za-z]', text) and re.search('\\d', text):
        return 'model_candidate'
    return 'unassigned_text'

def associate(detections, items, library, max_distance, left_penalty=0, allow_extended_resistor_refs=False):
    """No GT IDs, values, or pin labels are accepted by this function."""
    predictions = {d['id']: {'detection_id': d['id'], 'type': d['class_name'], 'designator': None, 'value': None, 'model': None, 'evidence': {}} for d in detections}
    typed = [{**item, 'role_candidate': classify(item, library)} for item in items]
    used = set()
    for field, role in [('designator', 'designator'), ('value', 'value'), ('model', 'model')]:
        proposals = []
        for det in detections:
            kind, box = (det['class_name'], det['body_bbox_xyxy'])
            if field == 'value' and kind not in ('r', 'c', 'l'):
                continue
            if field == 'model' and kind != 'box':
                continue
            for item in typed:
                eligible_role = item['role_candidate'] == role or (field == 'model' and item['role_candidate'] == 'model_candidate')
                extended=(allow_extended_resistor_refs and field=='designator' and kind=='r'
                          and re.fullmatch(r'R(?:[A-Z]\d+|[a-z])',item['text'].strip()) is not None)
                if extended and item.get('confidence',0)>=.95 and box_distance(box,item['bbox_xyxy'])<=20:
                    eligible_role=True
                if field == 'model' and item['text'] in library:
                    eligible_role = True
                if field == 'value' and kind == 'r' and (item['role_candidate'] == 'possible_pin_number') and re.fullmatch('\\d{3,4}', item['text'].strip()) and (predictions[det['id']]['designator'] is not None):
                    eligible_role = True
                if not eligible_role or item['id'] in used:
                    continue
                text, tb = (item['text'], item['bbox_xyxy'])
                # A newly supported connector prefix needs nearby, confident image text.
                if field == 'designator' and text.startswith('FPC'):
                    if kind != 'box' or item.get('confidence', 0) < .95 or box_distance(box, tb) > 25:
                        continue
                if field == 'model':
                    tx, ty = center(tb)
                    if box[0] <= tx <= box[2] and box[1] <= ty <= box[3]:
                        continue
                    if not (re.search('[A-Za-z]', text) and re.search('\\d', text)):
                        continue
                if field == 'designator' and (not text.startswith(PREFIX[kind])):
                    continue
                if field == 'designator' and kind == 'box':
                    tx, ty = center(tb)
                    if box[0] <= tx <= box[2] and box[1] <= ty <= box[3]:
                        continue
                if field == 'value':
                    suffix = text.lower()[-1:]
                    if kind == 'c' and suffix != 'f' or (kind == 'l' and suffix != 'h') or (kind == 'r' and suffix in ('f', 'h')):
                        continue
                distance = box_distance(box, tb)
                if distance > max_distance:
                    continue
                if kind == 'box':
                    alignment = abs(tb[0] - box[0]) * 0.15
                else:
                    a, b = (center(box), center(tb))
                    alignment = abs(a[1] - b[1]) * 0.25
                layout_cost = left_penalty if kind in ('r', 'c') and center(tb)[0] < center(box)[0] else 0
                score = distance + alignment + layout_cost
                proposals.append((score, det['id'], item['id'], item))
        occupied = set()
        for score, det_id, text_id, item in sorted(proposals, key=lambda p: p[:3]):
            if det_id in occupied or text_id in used:
                continue
            predictions[det_id][field] = item['text']
            if field == 'model':
                predictions[det_id]['library_status'] = 'exact' if item['text'] in library else 'unresolved'
            predictions[det_id]['evidence'][field] = {'ocr_id': text_id, 'association_cost': round(score, 3), 'ocr_confidence': item['confidence']}
            occupied.add(det_id)
            used.add(text_id)
    by_text={t['id']:t for t in items}
    for pred in predictions.values():
        evidence=pred['evidence'].get('model',{})
        token=by_text.get(evidence.get('ocr_id'),{})
        if pred.get('model') not in library and token.get('observed_green_wire_label'):
            pred['rejected_model_wire_evidence']={'text':pred['model'],'ocr_id':token['id'],'geometry':token['observed_green_wire_label']}
            pred['model']=None
            pred.pop('library_status',None)
            pred['evidence'].pop('model',None)
    return (list(predictions.values()), typed, sorted(used))
