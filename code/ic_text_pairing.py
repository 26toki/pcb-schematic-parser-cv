"""Pure image/text helpers. No training or GT entrypoint."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw

def midpoint(item):
    x1, y1, x2, y2 = item['bbox_xyxy']
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def pair_text(box, items, library_record, band, max_dy):
    """Pair outside numbers with inside labels, without reading GT pins."""
    x1, y1, x2, y2 = box
    middle = (x1 + x2) / 2
    pool = defaultdict(set)
    for pin in library_record.get('pins', []):
        pool[str(pin['number'])].add(pin['name'])
    numbers, names = ([], [])
    for item in items:
        x, y = midpoint(item)
        if not y1 <= y <= y2:
            continue
        token = item['text'].strip()
        is_number = token.isdigit() or (token in pool and re.fullmatch('[A-Za-z]+\\d+', token))
        if is_number:
            if x1 - band <= x < x1:
                numbers.append((item, 'left'))
            elif x2 < x <= x2 + band:
                numbers.append((item, 'right'))
        elif x1 <= x <= x2:
            if x <= middle and x - x1 <= band:
                names.append((item, 'left'))
            elif x > middle and x2 - x <= band:
                names.append((item, 'right'))
    proposals = []
    for number, side in numbers:
        for name, name_side in names:
            if side != name_side:
                continue
            token = number['text'].strip()
            if not token.isdigit() and name['text'] not in pool[token]:
                continue
            dy = abs(midpoint(number)[1] - midpoint(name)[1])
            if dy <= max_dy:
                proposals.append((dy, number['id'], name['id'], number, name, side))
    used_numbers, used_names, pairs = (set(), set(), [])
    for dy, number_id, name_id, number, name, side in sorted(proposals, key=lambda p: p[:3]):
        if number_id in used_numbers or name_id in used_names:
            continue
        used_numbers.add(number_id)
        used_names.add(name_id)
        num = number['text'].strip()
        candidates = sorted(pool.get(num, set()))
        pairs.append({'number': num, 'pinname': name['text'], 'side': side, 'number_ocr_id': number_id, 'name_ocr_id': name_id, 'row_difference_px': round(dy, 3), 'number_confidence': number['confidence'], 'name_confidence': name['confidence'], 'library_names': candidates, 'library_status': 'number_missing' if not candidates else 'exact_match' if name['text'] in candidates else 'name_conflict', 'pin_position': None})
    return (pairs, [item['id'] for item, _ in numbers if item['id'] not in used_numbers], [item['id'] for item, _ in names if item['id'] not in used_names])
