"""Frozen image-only detection helpers. No training/GT IO."""
import copy
def starts(length, size, stride):
    if length <= size:
        return [0]
    return sorted(set([*range(0, length - size + 1, stride), length - size]))

def refine(records, priors, width, height):
    result = copy.deepcopy(records)
    for item in result:
        choices = priors.get(item['class_name'])
        if not choices:
            continue
        x, y, u, v = item['body_bbox_xyxy']
        cx, cy = ((x + u) / 2, (y + v) / 2)
        choices2 = [s for s in choices if (s['height'] >= s['width']) == (v - y >= u - x)]
        chosen = (choices2 or choices)[0]
        w, h = (chosen['width'], chosen['height'])
        item['raw_bbox_xyxy'] = [x, y, u, v]
        item['body_bbox_xyxy'] = [max(0, cx - w / 2), max(0, cy - h / 2), min(width, cx + w / 2), min(height, cy + h / 2)]
        item['refinement'] = 'training_annotation_size_prior'
    return result

def predict(model, image, tiled=False, device=None):
    import torch
    from torchvision.ops import batched_nms
    w, h = image.size
    records = []
    windows = [(image, 0, 0, False)]
    if tiled and (w > 640 or h > 640):
        for y in starts(h, 640, 512):
            for x in starts(w, 640, 512):
                windows.append((image.crop((x, y, min(x + 640, w), min(y + 640, h))), x, y, True))
    for crop, x, y, is_tile in windows:
        selected_device = device if device is not None else 0 if torch.cuda.is_available() else 'cpu'
        result = model.predict(crop, imgsz=640, device=selected_device, conf=0.25, iou=0.7, max_det=1000, verbose=False)[0]
        for b in result.boxes:
            q = b.xyxy[0].cpu().tolist()
            if is_tile and (x > 0 and q[0] <= 2 or (y > 0 and q[1] <= 2) or (x + crop.width < w and q[2] >= crop.width - 2) or (y + crop.height < h and q[3] >= crop.height - 2)):
                continue
            records.append(dict(class_id=int(b.cls.item()), class_name=result.names[int(b.cls.item())], body_bbox_xyxy=[q[0] + x, q[1] + y, q[2] + x, q[3] + y], confidence=float(b.conf.item()), source='detector', view='tile' if is_tile else 'full'))
    if tiled and records:
        keep = batched_nms(torch.tensor([r['body_bbox_xyxy'] for r in records]), torch.tensor([r['confidence'] for r in records]), torch.tensor([r['class_id'] for r in records]), 0.5).tolist()
        records = [records[i] for i in keep]
    for i, r in enumerate(records):
        r['id'] = f'detection_{i + 1:04d}'
    return records

def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0

def merge_refined(records):
    kept = []
    for r in sorted(records, key=lambda r: -r['confidence']):
        if not any((k['class_name'] == r['class_name'] and iou(k['body_bbox_xyxy'], r['body_bbox_xyxy']) > 0.5 for k in kept)):
            kept.append(copy.deepcopy(r))
    for i, r in enumerate(kept):
        r['id'] = f'detection_{i + 1:04d}'
    return kept

def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

def contained_fraction(child, parent):
    inter = max(0, min(child[2], parent[2]) - max(child[0], parent[0])) * max(0, min(child[3], parent[3]) - max(child[1], parent[1]))
    return inter / max(area(child), 1e-09)

def filter_candidates(candidates, full_detections):
    anchors = [d for d in full_detections if d['class_name'] == 'box' and d['confidence'] >= 0.8]
    kept = []
    suppressed = []
    flagged = []
    for original in candidates:
        d = copy.deepcopy(original)
        b = d['body_bbox_xyxy']
        parents = [p for p in anchors if contained_fraction(b, p['body_bbox_xyxy']) >= 0.9 and area(b) / max(area(p['body_bbox_xyxy']), 1e-09) <= 0.7 and (p['confidence'] >= d['confidence'])]
        if d['class_name'] == 'box' and parents:
            parent = max(parents, key=lambda p: p['confidence'])
            suppressed.append(dict(candidate=d, parent=parent, reason='smaller_lower_confidence_box_inside_high_confidence_full_image_box'))
            continue
        if d['class_name'] != 'box' and parents:
            d['candidate_review_reasons'] = ['component_inside_high_confidence_ic_body']
            flagged.append(d['id'])
        kept.append(d)
    return (kept, suppressed, flagged)
