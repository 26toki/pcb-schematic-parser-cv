"""Recover visually observed numeric pins using both sides of the lead row; no GT or LLM."""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageOps
from rapidocr_onnxruntime import RapidOCR
from locate_detector_ic_pins import observe_contact


def agree(attempts,min_confidence=.85):
    supported=[a for a in attempts if a['raw'].strip().isdigit() and a['confidence']>=min_confidence]
    by_side=defaultdict(list)
    for a in supported:by_side[a['vertical_crop']].append(a)
    choices=[]
    for side,items in by_side.items():
        if len({a['width'] for a in items})<2 or len({a['raw'].strip() for a in items})!=1:continue
        choices.append(max(items,key=lambda a:a['confidence']))
    if len({a['raw'].strip() for a in choices})!=1:return None
    return choices[0] if choices else None


def merge_visible(existing,accepted):
    """Reject collisions first; never delete an old row because a rejected proposal covered it."""
    def covers(old,new):
        return old.get('body_contact_xy') is not None and old['pinname']==new['pinname'] and old['side']==new['side'] and math.dist(old['body_contact_xy'],new['body_contact_xy'])<=4
    while True:
        preserved=[old for old in existing if not any(covers(old,new) for new in accepted)]
        valid=[new for new in accepted if not any(old['number']==new['number'] for old in preserved)]
        if len(valid)==len(accepted):break
        accepted=valid
    return preserved+accepted,[(new,[old for old in existing if covers(old,new)]) for new in accepted]


def library_supported_name(number,name,record):
    names=sorted({p['name'] for p in record.get('pins',[]) if str(p['number'])==number})
    if name in names:return name,names
    matches=[n for n in names if n.casefold()==name.casefold()]
    return (matches[0],names) if len(matches)==1 else (None,names)


def strong_visual_evidence(pin):
    """Library identity alone cannot validate which adjacent row supplied a digit."""
    return (pin.get('library_status')=='exact_match' and pin.get('text_source') in
            ('two_width_consensus_above_or_below_observed_lead','rotated_local_ocr_library_and_vertical_lead_support'))


def merge_library_visible(existing,accepted):
    """Protect independent visual consensus; allow coherent repairs of weaker pairs."""
    protected=[p for p in existing if strong_visual_evidence(p)]
    allowed=[]
    for new in accepted:
        conflicts=any(old['number']==new['number'] or
                      (old.get('body_contact_xy') is not None and old['side']==new['side'] and
                       math.dist(old['body_contact_xy'],new['body_contact_xy'])<=4)
                      for old in protected)
        if not conflicts:allowed.append(new)
    return merge_visible(existing,allowed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pins','local-ocr','library','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    source=json.loads(a.pins.read_text(encoding='utf-8'));output=copy.deepcopy(source)
    library=json.loads(a.library.read_text(encoding='utf-8-sig'))
    image=Image.open(source['image']).convert('RGB');mask=np.asarray(image.convert('L'))<160
    engine=RapidOCR();attempts=[];events=[]
    for ic in output['ics']:
        record=library.get(ic['model'],{});known=bool(record.get('pins'))
        items=json.loads((a.local_ocr/ic['detection_id']/'ocr.json').read_text(encoding='utf-8'))['items']
        texts={item['id']:item for item in items};box=ic['body_bbox_xyxy'];proposals=[];eligible=[]
        for item in items:
            name=item['text'];x1,y1,x2,y2=item['bbox_xyxy'];cx,cy=(x1+x2)/2,(y1+y2)/2
            if item['confidence']<.85 or not name.strip() or name.strip().isdigit():continue
            if not (box[0]+2<cx<box[2]-2 and box[1]+2<cy<box[3]-2):continue
            side='left' if cx-box[0]<box[2]-cx else 'right'
            if min(cx-box[0],box[2]-cx)>100:continue
            contact=observe_contact(mask,box,dict(side=side,name_ocr_id=item['id']),texts)
            if contact['body_contact_xy'] is None:continue
            if known and any(strong_visual_evidence(p) and p['side']==side and
                             p.get('body_contact_xy') is not None and math.dist(p['body_contact_xy'],contact['body_contact_xy'])<=4
                             for p in ic['pins']):continue
            eligible.append((item,side,contact))
        for item,side,contact in eligible:
            name=item['text'];x1,y1,x2,y2=item['bbox_xyxy']
            edge,row=contact['body_contact_xy'];font=max(5,y2-y1);height=max(10,min(32,math.ceil(font*1.7)))
            widths=[24,max(30,min(60,math.ceil(font*2.5)))];readings=[]
            for vertical in ('above','below'):
                for width in widths:
                    region=[max(0,edge-width if side=='left' else edge+2),
                            max(0,row-height if vertical=='above' else row+2),
                            min(image.width,edge-2 if side=='left' else edge+width),
                            min(image.height,row+3 if vertical=='above' else row+height)]
                    if region[0]>=region[2] or region[1]>=region[3]:continue
                    patch=np.asarray(image.crop(region)).copy()
                    chroma=patch.max(axis=2).astype(int)-patch.min(axis=2).astype(int);patch[chroma>40]=255
                    black=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)<160
                    count,_,stats,_=cv2.connectedComponentsWithStats(black.astype(np.uint8),8)
                    glyphs=[]
                    for j in range(1,count):
                        gx,gy,gw,gh,area=map(int,stats[j])
                        if gh<5 or gh>font*1.6 or area<5:continue
                        glyph_y=region[1]+gy+gh/2
                        other_rows=[p['body_contact_xy'][1] for _,s,p in eligible if s==side]
                        if min(abs(glyph_y-r) for r in other_rows)+1<abs(glyph_y-row):continue
                        glyphs.append((gx,gy,gx+gw,gy+gh))
                    if not glyphs:continue
                    # Crop to the nearest observed row's black glyphs, with white margin.
                    tight=[min(g[0] for g in glyphs),min(g[1] for g in glyphs),max(g[2] for g in glyphs),max(g[3] for g in glyphs)]
                    chip=Image.fromarray(patch).crop(tight);scale=64/chip.height
                    chip=ImageOps.expand(chip.resize((max(1,round(chip.width*scale)),64),Image.Resampling.LANCZOS),border=16,fill='white')
                    file=a.output/'number_crops'/f'{ic["detection_id"]}_{item["id"]}_{vertical}_{width}.png'
                    file.parent.mkdir(exist_ok=True);chip.save(file)
                    result,_=engine(str(file),use_det=False,use_cls=False,use_rec=True)
                    raw,confidence=result[0][:2] if result else ('',0)
                    reading=dict(ic_id=ic['detection_id'],name=name,raw=raw,confidence=float(confidence),
                                 vertical_crop=vertical,width=width,crop_xyxy=region,crop_path=str(file.resolve()))
                    readings.append(reading);attempts.append(reading)
            best=agree(readings)
            if best is None:continue
            number=best['raw'].strip();validated_name=name;library_names=[]
            if known:
                validated_name,library_names=library_supported_name(number,name,record)
                if validated_name is None:continue
            proposals.append(dict(number=number,pinname=validated_name,raw_pinname=name,side=side,name_ocr_id=item['id'],
                  number_confidence=best['confidence'],name_confidence=item['confidence'],body_contact_xy=[edge,row],
                  pin_position=contact['pin_position'],position_status=contact['position_status'],
                  library_status='exact_match' if known else 'model_not_in_library_visual_evidence_only',library_names=library_names,
                  text_source='two_width_consensus_above_or_below_observed_lead',number_crop_xyxy=best['crop_xyxy']))
        grouped=defaultdict(list)
        for pin in proposals:grouped[pin['number']].append(pin)
        accepted=[]
        for number,options in grouped.items():
            if len({(p['pinname'],tuple(p['body_contact_xy'])) for p in options})!=1:
                events.append(dict(ic_id=ic['detection_id'],number=number,status='ambiguous_rows_or_names'));continue
            accepted.append(options[0])
        # Replace covered label/row records only as a coherent unique-number set, preserving audit evidence.
        merged,changes=(merge_library_visible if known else merge_visible)(ic['pins'],accepted)
        for pin,before in changes:
            status='added_visual_candidate' if not before else 'replaced_by_visual_consensus' if any(old['number']!=pin['number'] for old in before) else 'reconfirmed_visible_pair'
            events.append(dict(ic_id=ic['detection_id'],number=pin['number'],pinname=pin['pinname'],status=status,before=before))
        ic['pins']=merged
    output['summary'].update(text_pairs=sum(len(ic['pins']) for ic in output['ics']),
        observed_contacts=sum(pin['body_contact_xy'] is not None for ic in output['ics'] for pin in ic['pins']),
        free_endpoint_candidates=sum(pin['pin_position'] is not None for ic in output['ics'] for pin in ic['pins']))
    output['visible_pin_recovery']=dict(events=events,attempts=attempts,input_sha256=hashlib.sha256(a.pins.read_bytes()).hexdigest(),
        limitations='Numeric left/right pins with >=.85 confidence and two-width agreement in same above/below region. Known models also require library name/number support and preserve existing exact records. No ambiguous I/l to 1 substitution, no library or LLM invented pins. Added records remain unverified predictions.')
    (a.output/'pins.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    draw=ImageDraw.Draw(image)
    for ic in output['ics']:
        for pin in ic['pins']:
            if pin['text_source']=='two_width_consensus_above_or_below_observed_lead':
                x,y=pin['body_contact_xy'];draw.ellipse((x-3,y-3,x+3,y+3),outline='green',width=2);draw.text((x+4,y),pin['number']+':'+pin['pinname'],fill='green')
    image.save(a.output/'recovered_contacts.png');print(json.dumps(dict(Counter(e['status'] for e in events))))


if __name__=='__main__':main()
