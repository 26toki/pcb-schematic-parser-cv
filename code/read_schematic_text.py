"""Lesson 7: CPU OCR baseline; compare full image and a manual crop, without GT."""
import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw
from rapidocr_onnxruntime import RapidOCR


def recognize(engine, original, region, scale, folder):
    left, top, right, bottom = region
    patch = original.crop(region)
    resized = patch.resize((round(patch.width*scale), round(patch.height*scale)), Image.Resampling.LANCZOS)
    folder.mkdir(parents=True, exist_ok=True)
    input_path = folder / 'ocr_input.png'
    resized.save(input_path)
    begin = time.perf_counter()
    result, _ = engine(str(input_path))
    elapsed = time.perf_counter()-begin
    records = []
    overlay = original.copy()
    draw = ImageDraw.Draw(overlay)
    sx, sy = resized.width/patch.width, resized.height/patch.height
    for index, (quad, text, score) in enumerate(result or [], 1):
        points = [[float(x)/sx+left, float(y)/sy+top] for x, y in quad]
        xs, ys = zip(*points)
        box = [min(xs), min(ys), max(xs), max(ys)]
        record = {'id': f'text_{index:03d}', 'text': text, 'confidence': float(score),
                  'polygon': points, 'bbox_xyxy': box}
        records.append(record)
        draw.line([tuple(p) for p in points]+[tuple(points[0])], fill=(0, 100, 240), width=2)
        draw.text((box[0], max(0, box[1]-12)), str(index), fill=(220, 20, 20))
    overlay.save(folder / 'overlay.png')
    report = {'coordinate_system': 'original image pixels, top-left origin', 'crop': list(region),
              'requested_scale': scale, 'elapsed_seconds': round(elapsed, 3),
              'note': 'OCR scores are not calibrated correctness probabilities; no semantic assignment yet.',
              'items': records}
    (folder / 'ocr.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (folder / 'texts.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'text', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        for item in records:
            writer.writerow([item['id'], item['text'], round(item['confidence'], 4), *item['bbox_xyxy']])
    return {'regions': len(records), 'elapsed_seconds': round(elapsed, 3),
            'texts': [item['text'] for item in records]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--crop', nargs=4, type=int, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    parser.add_argument('--full-only', action='store_true', help='Skip manual crop experiments for batch diagnostics')
    parser.add_argument('--crop-scale', type=float, default=2)
    args = parser.parse_args()
    rgba = Image.open(args.image).convert('RGBA')
    original = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
    width, height = original.size
    if args.crop is None and not args.full_only:
        parser.error('--crop is required unless --full-only is used')
    crop = args.crop or (0, 0, width, height)
    left, top, right, bottom = crop
    if not (0 <= left < right <= width and 0 <= top < bottom <= height and 0.1 <= args.crop_scale <= 4):
        parser.error('invalid crop or scale; scale must be 0.1..4')
    engine = RapidOCR()
    # Keep the same crop at two scales to separate crop effects from scale effects.
    summary = {'full': recognize(engine, original, (0, 0, width, height), 1, args.output/'full')}
    if not args.full_only:
        summary['crop_1x'] = recognize(engine, original, crop, 1, args.output/'crop_1x')
        summary['crop_scaled'] = recognize(engine, original, crop, args.crop_scale, args.output/'crop_scaled')
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
