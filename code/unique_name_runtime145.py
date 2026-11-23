"""Bounded unique-library-number recovery from visible names and leads."""
import math,json,collections
import numpy as np
from PIL import Image,ImageOps
from library_pin_rules144 import unique_numbers,agreed_name
from locate_detector_ic_pins import observe_contact
from locate_symbol_pin_tips import color_pin_tip
def recover(output,image,library,full_items,local_root,engine):
    mask=np.asarray(image.convert('L'))<160;rgb=np.asarray(image.convert('RGB'));attempts=[];events=[];budget=12;ocr_calls=0
    for ic in output['ics']:
        if not ic.get('designator'):continue
        mapping=unique_numbers(library.get(ic['model'],{}));existing={p['number'] for p in ic['pins']};mapping={k:v for k,v in mapping.items() if v not in existing}
        path=local_root/ic['detection_id']/'ocr.json';items=full_items+(json.loads(path.read_text(encoding='utf-8'))['items'] if path.exists() else []);box=ic['body_bbox_xyxy'];seen=[];proposals=[]
        for t in items:
            name=t['text'].strip();b=t['bbox_xyxy'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
            if name not in mapping or t['confidence']<.9 or not(box[0]+2<cx<box[2]-2 and box[1]+2<cy<box[3]-2):continue
            if any(n==name and math.dist(p,[cx,cy])<5 for n,p in seen):continue
            seen.append((name,[cx,cy]));side='left' if cx-box[0]<box[2]-cx else 'right';contact=observe_contact(mask,box,dict(side=side,name_ocr_id='name'),{'name':t})
            if contact['body_contact_xy'] is None or len(attempts)>=budget:continue
            region=[max(0,math.floor(b[0])-2),max(0,math.floor(b[1])-2),min(image.width,math.ceil(b[2])+2),min(image.height,math.ceil(b[3])+2)];crop=image.crop(region);votes=[]
            for zoom in [3,4]:
                patch=ImageOps.expand(crop.resize((crop.width*zoom,crop.height*zoom),Image.Resampling.LANCZOS),border=12,fill='white')
                ocr_calls+=1;raw,_=engine(np.asarray(patch),use_det=False,use_cls=False,use_rec=True);votes.append([dict(text=s.strip(),confidence=float(c)) for s,c in raw or []])
            evidence=dict(ic_id=ic['detection_id'],name=name,number=mapping[name],votes=votes,crop_xyxy=region);attempts.append(evidence)
            if not agreed_name(name,votes):
                from text_glyph_render146 import render
                tight=[]
                for height in [64,80]:
                    chip=render(crop,height)
                    raw=None
                    if chip is not None:
                        ocr_calls+=1
                        raw,_=engine(np.asarray(chip),use_det=False,use_cls=False,use_rec=True)
                    tight.append([dict(text=t.strip(),confidence=float(c)) for t,c in raw or []])
                evidence['tight_votes']=tight
                if not all(len(v)==1 and v[0]['text']==name and v[0]['confidence']>=.95 for v in tight):
                    from soft_name170 import render as soft_render,accept
                    soft=[]
                    for height in (32,48):
                        chip=soft_render(crop,height);raw=None
                        if chip is not None:
                            ocr_calls+=1
                            raw,_=engine(np.asarray(chip),use_det=False,use_cls=False,use_rec=True)
                        soft.append([dict(text=t.strip(),confidence=float(c)) for t,c in raw or []])
                    evidence['soft_votes']=soft
                    if not accept(name,soft):continue
                    tight=soft
                votes=tight
            pin=dict(number=mapping[name],pinname=name,side=side,body_contact_xy=contact['body_contact_xy'],pin_position=None,library_status='exact_match',number_source='unique_library_mapping_for_visible_name',text_source='two_view_name_unique_library_number',name_confidence=min(v[0]['confidence'] for v in votes))
            tip=color_pin_tip(rgb,pin)
            if tip is None:continue
            pin.update(pin_position=tip['position_image_xy'],tip_geometry=tip,position_status='observed_color_boundary')
            proposals.append(pin)
        grouped=collections.defaultdict(list)
        for p in proposals:grouped[p['number']].append(p)
        for number,options in grouped.items():
            if len({(p['pinname'],tuple(p['body_contact_xy'])) for p in options})!=1:continue
            p=options[0]
            if any(q.get('body_contact_xy') is not None and math.dist(p['body_contact_xy'],q['body_contact_xy'])<=3 for q in ic['pins']):continue
            ic['pins'].append(p);events.append(dict(ic_id=ic['detection_id'],designator=ic['designator'],number=number,name=p['pinname']))
    output['unique_library_pin_recovery']=dict(events=events,attempts=attempts,max_attempts=budget,OCR_calls=ocr_calls)
    output['summary'].update(text_pairs=sum(len(ic['pins']) for ic in output['ics']),observed_contacts=sum(p.get('body_contact_xy') is not None for ic in output['ics'] for p in ic['pins']),free_endpoint_candidates=sum(p.get('pin_position') is not None for ic in output['ics'] for p in ic['pins']))
    return output
