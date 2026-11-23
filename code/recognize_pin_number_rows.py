"""Read tiny numeric pin labels above observed wire rows, without GT geometry or labels."""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from rapidocr_onnxruntime import RapidOCR
from locate_detector_ic_pins import observe_contact


def supported_number(raw,name,pool):
    clean=raw.strip('.,，。 ')
    correction=None if clean==raw else 'edge_punctuation_removed'
    if clean in ('I','l','L') and name in pool.get('1',set()):
        clean='1';correction='ambiguous_single_glyph_to_one_with_library_name_support'
    if clean.isdigit() and name in pool.get(clean,set()):return clean,correction
    return None,None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','library','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--cases',nargs='+',help='Case folder names; default retains the ten-case diagnostic batch')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    library=json.loads(a.library.read_text(encoding='utf-8-sig'));engine=RapidOCR();totals=Counter()
    for case in (a.cases or [str(i) for i in range(1,11)]):
        source=a.source/str(case)/'pins.json';report=json.loads(source.read_text(encoding='utf-8'))
        image=Image.open(report['image']).convert('RGB');mask=np.asarray(image.convert('L'))<160
        output=copy.deepcopy(report);attempts=[];events=[];dest=a.output/str(case);dest.mkdir(parents=True,exist_ok=True)
        for ic in output['ics']:
            folder=a.source/str(case)/'local_ocr'/ic['detection_id']
            items=json.loads((folder/'ocr.json').read_text(encoding='utf-8'))['items']
            for item in items:item['id']=ic['detection_id']+'_local_'+item['id']
            texts={t['id']:t for t in items}
            pool=defaultdict(set)
            for pin in library.get(ic['model'],{}).get('pins',[]):pool[str(pin['number'])].add(pin['name'])
            names={n for v in pool.values() for n in v};proposals=[];box=ic['body_bbox_xyxy']
            for item in items:
                raw_name=item['text'];name=raw_name
                if name not in names:
                    matches=[n for n in names if n.casefold()==name.casefold()]
                    if len(matches)!=1:continue
                    name=matches[0]
                x1,y1,x2,y2=item['bbox_xyxy'];cx,cy=(x1+x2)/2,(y1+y2)/2
                if not (box[0]<=cx<=box[2] and box[1]<=cy<=box[3]):continue
                side='left' if cx-box[0]<box[2]-cx else 'right'
                if min(cx-box[0],box[2]-cx)>100:continue
                pseudo=dict(side=side,name_ocr_id=item['id'])
                observed=observe_contact(mask,box,pseudo,texts)
                if observed['body_contact_xy'] is None:continue
                edge,row=observed['body_contact_xy'];font_height=max(5,y2-y1)
                crop_height=max(10,min(40,math.ceil(font_height*1.7)))
                widths=sorted({24,max(24,min(64,math.ceil(font_height*2.5)))})
                valid=[]
                for width in widths:
                    region=[max(0,edge-width if side=='left' else edge+2),max(0,row-crop_height),
                            min(image.width,edge-2 if side=='left' else edge+width),max(0,row-2)]
                    if region[0]>=region[2] or region[1]>=region[3]:continue
                    patch=np.asarray(image.crop(region)).copy()
                    chroma=patch.max(axis=2).astype(int)-patch.min(axis=2).astype(int);patch[chroma>40]=255
                    chip=Image.fromarray(patch);scale=80/chip.height
                    chip=chip.resize((max(1,round(chip.width*scale)),80),Image.Resampling.LANCZOS)
                    chip=ImageOps.expand(chip,border=12,fill='white')
                    namefile=f'{ic["detection_id"]}_{item["id"]}_w{width}.png';file=dest/'number_crops'/namefile
                    file.parent.mkdir(exist_ok=True);chip.save(file)
                    result,_=engine(str(file),use_det=False,use_cls=False,use_rec=True)
                    raw,confidence=result[0][:2] if result else ('',0)
                    number,correction=supported_number(raw,name,pool)
                    attempt=dict(ic_id=ic['detection_id'],raw_name=raw_name,validated_name=name,raw_number=raw,
                                 confidence=float(confidence),supported_number=number,correction=correction,
                                 contact_xy=[edge,row],crop_xyxy=region,crop_path=str(file.resolve()))
                    attempts.append(attempt)
                    if number is not None and confidence>=.7:valid.append(attempt)
                if len({v['supported_number'] for v in valid})!=1:continue
                best=max(valid,key=lambda v:v['confidence']);number=best['supported_number']
                proposals.append(dict(number=number,pinname=name,raw_pinname=raw_name,side=side,
                                      name_ocr_id=item['id'],raw_number=best['raw_number'],number_confidence=best['confidence'],
                                      name_confidence=item['confidence'],library_names=sorted(pool[number]),library_status='exact_match',
                                      text_source='observed_wire_row_recognizer_only',number_crop_xyxy=best['crop_xyxy'],
                                      correction=best['correction'],body_contact_xy=[edge,row],pin_position=observed['pin_position'],
                                      position_status=observed['position_status']))
            grouped=defaultdict(list)
            for pin in proposals:grouped[pin['number']].append(pin)
            for number,options in grouped.items():
                if len({tuple(p['body_contact_xy']) for p in options})>1:
                    events.append(dict(ic_id=ic['detection_id'],number=number,action='ambiguous_multiple_rows_not_accepted'));continue
                candidate=max(options,key=lambda p:p['number_confidence'])
                existing=[p for p in ic['pins'] if p['number']==number]
                if not existing:
                    ic['pins'].append(candidate);events.append(dict(ic_id=ic['detection_id'],number=number,action='added'))
                elif all(p.get('library_status')!='exact_match' for p in existing):
                    ic['pins']=[p for p in ic['pins'] if p['number']!=number]+[candidate]
                    events.append(dict(ic_id=ic['detection_id'],number=number,action='replaced_library_conflict',before=existing))
        output['summary'].update(text_pairs=sum(len(ic['pins']) for ic in output['ics']),
                                 observed_contacts=sum(p['body_contact_xy'] is not None for ic in output['ics'] for p in ic['pins']),
                                 free_endpoint_candidates=sum(p['pin_position'] is not None for ic in output['ics'] for p in ic['pins']))
        output['row_recognition']=dict(attempts=attempts,events=events,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        (dest/'pins.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
        draw=ImageDraw.Draw(image)
        for ic in output['ics']:
            draw.rectangle(ic['body_bbox_xyxy'],outline='blue',width=2)
            for pin in ic['pins']:
                if pin['body_contact_xy']:
                    x,y=pin['body_contact_xy'];draw.ellipse((x-3,y-3,x+3,y+3),outline='red',width=2)
        image.save(dest/'observed_pins.png')
        counts=Counter(e['action'] for e in events);totals.update(counts)
        print(case,json.dumps(dict(counts)),flush=True)
    (a.output/'recognition_summary.json').write_text(json.dumps(dict(totals),indent=2),encoding='utf-8')


if __name__=='__main__':main()
