"""Rotate library-supported vertical IC leads into horizontal OCR coordinates; no GT."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageOps
from rapidocr_onnxruntime import RapidOCR
from read_schematic_text import recognize
from ic_text_pairing import pair_text
from locate_detector_ic_pins import observe_contact


def row_number(engine,image,edge,row,side,font,rows,allowed,folder):
    readings=[];height=max(10,min(32,math.ceil(font*1.7)))
    for direction in ('above','below'):
        for width in (24,max(30,min(60,math.ceil(font*2.5)))):
            crop=[max(0,edge-width if side=='left' else edge+2),max(0,row-height if direction=='above' else row+2),
                  min(image.width,edge-2 if side=='left' else edge+width),min(image.height,row+3 if direction=='above' else row+height)]
            if crop[0]>=crop[2] or crop[1]>=crop[3]:continue
            patch=np.asarray(image.crop(crop)).copy();chroma=patch.max(axis=2).astype(int)-patch.min(axis=2).astype(int);patch[chroma>40]=255
            # Small rotated digits have pale antialias pixels connecting their strokes.
            mask=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)<220
            count,_,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8);glyphs=[]
            for i in range(1,count):
                x,y,w,h,area=map(int,stats[i]);cy=crop[1]+y+h/2
                if h>=4 and h<=font*1.6 and area>=5 and min(abs(cy-r) for r in rows)+1>=abs(cy-row):glyphs.append((x,y,x+w,y+h))
            if not glyphs:continue
            chip=Image.fromarray(patch).crop([min(g[0] for g in glyphs),min(g[1] for g in glyphs),max(g[2] for g in glyphs),max(g[3] for g in glyphs)])
            chip=ImageOps.expand(chip.resize((max(1,round(chip.width*64/chip.height)),64),Image.Resampling.LANCZOS),border=16,fill='white')
            folder.mkdir(parents=True,exist_ok=True);file=folder/(direction+'_'+str(width)+'.png');chip.save(file)
            result,_=engine(str(file),use_det=False,use_cls=False,use_rec=True);raw,confidence=result[0][:2] if result else ('',0)
            number=raw.strip('.,，。 _');correction=None
            if number in ('I','l','L') and '1' in allowed:number='1';correction='single_glyph_with_library_name_support'
            readings.append(dict(raw=raw,number=number,confidence=float(confidence),direction=direction,width=width,
                                 crop_xyxy=crop,correction=correction,crop_path=str(file.resolve())))
    accepted=[]
    for direction in ('above','below'):
        supported=[r for r in readings if r['direction']==direction and r['confidence']>=.8 and r['number'] in allowed]
        if len({r['width'] for r in supported})>=2 and len({r['number'] for r in supported})==1:accepted.append(max(supported,key=lambda r:r['confidence']))
    return (max(accepted,key=lambda r:r['confidence']) if accepted and len({r['number'] for r in accepted})==1 else None),readings


def rotated_point(xy,width,height,clockwise):
    x,y=xy
    return [height-1-y,x] if clockwise==90 else [y,width-1-x]


def restored_point(xy,width,height,clockwise):
    x,y=xy
    return [y,height-1-x] if clockwise==90 else [width-1-y,x]


def rotated_box(box,width,height,clockwise):
    x,y,u,v=box;points=[rotated_point(p,width,height,clockwise) for p in ([x,y],[x,v],[u,y],[u,v])]
    return [min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]


def lead_groups(mask,box):
    x,y,u,v=map(int,box);found=[]
    for side,edge,sign in [('top',y,-1),('bottom',v,1)]:
        xs=[]
        for xx in range(x+4,u-3):
            yy=[edge+sign*k for k in range(3,12)]
            if min(yy)<0 or max(yy)>=mask.shape[0]:continue
            if mask[yy,max(0,xx-1):xx+2].any(axis=1).mean()>=.9:xs.append(xx)
        groups=[]
        for xx in xs:
            if not groups or xx>groups[-1][-1]+1:groups.append([xx])
            else:groups[-1].append(xx)
        found.extend(dict(side=side,x=float(np.mean(g)),edge=edge) for g in groups if len(g)<=8)
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pins','library','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    source=json.loads(a.pins.read_text(encoding='utf-8'));output=copy.deepcopy(source)
    library=json.loads(a.library.read_text(encoding='utf-8-sig'));image=Image.open(source['image']).convert('RGB')
    mask=np.asarray(image.convert('L'))<160;engine=None;events=[];attempts=[];number_attempts=[]
    for ic in output['ics']:
        record=library.get(ic['model'],{});leads=lead_groups(mask,ic['body_bbox_xyxy'])
        if not record.get('pins') or len(leads)<2:continue
        if engine is None:engine=RapidOCR()
        b=ic['body_bbox_xyxy'];region=[max(0,b[0]-45),max(0,b[1]-45),min(image.width,b[2]+45),min(image.height,b[3]+45)]
        patch=image.crop(region);local=[b[0]-region[0],b[1]-region[1],b[2]-region[0],b[3]-region[1]];proposals=[]
        for angle in (90,270):
            rotated=patch.transpose(Image.Transpose.ROTATE_270 if angle==90 else Image.Transpose.ROTATE_90)
            box=rotated_box(local,patch.width,patch.height,angle);folder=a.output/'rotated_ocr'/ic['detection_id']/str(angle)
            recognize(engine,rotated,[0,0,rotated.width,rotated.height],3,folder)
            items=json.loads((folder/'ocr.json').read_text(encoding='utf-8'))['items'];texts={item['id']:item for item in items}
            pairs,_,_=pair_text(box,items,record,60,16);rotmask=np.asarray(rotated.convert('L'))<160
            pool=defaultdict(set)
            for p in record['pins']:pool[p['name']].add(str(p['number']))
            eligible=[]
            for item in items:
                name=item['text'];x1,y1,x2,y2=item['bbox_xyxy'];cx,cy=(x1+x2)/2,(y1+y2)/2
                if name not in pool or item['confidence']<.85 or not (box[0]<cx<box[2] and box[1]<cy<box[3]):continue
                side='left' if cx-box[0]<box[2]-cx else 'right'
                observed=observe_contact(rotmask,box,dict(side=side,name_ocr_id=item['id']),texts)
                if observed['body_contact_xy'] is not None:eligible.append((item,side,observed))
            for item,side,observed in eligible:
                edge,row=observed['body_contact_xy'];font=item['bbox_xyxy'][3]-item['bbox_xyxy'][1]
                best,readings=row_number(engine,rotated,edge,row,side,font,
                    [p['body_contact_xy'][1] for _,s,p in eligible if s==side],pool[item['text']],folder/'numbers'/item['id'])
                number_attempts.extend(dict(ic_id=ic['detection_id'],angle=angle,name=item['text'],**r) for r in readings)
                if best is not None:
                    pairs.append(dict(number=best['number'],pinname=item['text'],side=side,name_ocr_id=item['id'],
                        number_confidence=best['confidence'],name_confidence=item['confidence'],library_status='exact_match',
                        library_names=[item['text']],numeric_source='two_width_observed_row',number_correction=best['correction']))
            # Pair detected numeric glyphs by observed wire row, not name-text midpoint.
            for item in items:
                number=item['text'].strip();x1,y1,x2,y2=item['bbox_xyxy'];cx,cy=(x1+x2)/2,(y1+y2)/2
                if not number.isdigit() or item['confidence']<.7 or not box[1]<=cy<=box[3]:continue
                side='left' if box[0]-30<=cx<box[0] else 'right' if box[2]<cx<=box[2]+30 else None
                options=sorted([(abs(cy-observed['body_contact_xy'][1]),name,observed) for name,s,observed in eligible if s==side],key=lambda option:option[0])
                if not options or options[0][0]>14 or (len(options)>1 and options[1][0]-options[0][0]<3):continue
                _,name,observed=options[0]
                if number not in pool[name['text']]:continue
                pairs.append(dict(number=number,pinname=name['text'],side=side,name_ocr_id=name['id'],
                    number_confidence=item['confidence'],name_confidence=name['confidence'],library_status='exact_match',
                    library_names=[name['text']],numeric_source='detected_glyph_nearest_observed_row_with_library_support'))
            for pair in pairs:
                minimum=.7 if pair.get('numeric_source')=='detected_glyph_nearest_observed_row_with_library_support' else .8
                if pair['library_status']!='exact_match' or pair['number_confidence']<minimum or pair['name_confidence']<.85:continue
                observed=observe_contact(rotmask,box,pair,texts)
                if observed['body_contact_xy'] is None:continue
                xy=restored_point(observed['body_contact_xy'],patch.width,patch.height,angle)
                xy=[xy[0]+region[0],xy[1]+region[1]]
                side='top' if abs(xy[1]-b[1])<abs(xy[1]-b[3]) else 'bottom'
                if not any(lead['side']==side and abs(lead['x']-xy[0])<=4 for lead in leads):continue
                pin={**pair,'side':side,'body_contact_xy':xy,'pin_position':None,
                     'position_status':'vertical_body_contact_tip_unresolved',
                     'text_source':'rotated_local_ocr_library_and_vertical_lead_support',
                     'rotation_clockwise':angle,'source_crop_xyxy':region}
                proposals.append(pin)
            attempts.append(dict(ic_id=ic['detection_id'],angle=angle,raw_pairs=len(pairs),cumulative_accepted_rotation_pairs=len(proposals)))
        grouped=defaultdict(list)
        for pin in proposals:grouped[pin['number']].append(pin)
        for number,options in grouped.items():
            first=options[0]
            if any(p['pinname']!=first['pinname'] or math.dist(p['body_contact_xy'],first['body_contact_xy'])>4 for p in options):
                events.append(dict(ic_id=ic['detection_id'],number=number,status='conflicting_rotation_candidates'));continue
            if any(p['number']==number for p in ic['pins']):continue
            best=max(options,key=lambda p:p['number_confidence']+p['name_confidence']);ic['pins'].append(best)
            events.append(dict(ic_id=ic['detection_id'],number=number,pinname=best['pinname'],status='added_vertical_candidate'))
    output['summary'].update(text_pairs=sum(len(ic['pins']) for ic in output['ics']),
        observed_contacts=sum(pin['body_contact_xy'] is not None for ic in output['ics'] for pin in ic['pins']),
        free_endpoint_candidates=sum(pin['pin_position'] is not None for ic in output['ics'] for pin in ic['pins']))
    output['vertical_pin_recovery']=dict(events=events,attempts=attempts,number_attempts=number_attempts,input_sha256=hashlib.sha256(a.pins.read_bytes()).hexdigest(),
        limitations='Top/bottom numeric pins with library-exact OCR and visible vertical leads. No GT, inferred library pins or API. Rotated coordinates restored to full image. Added records remain unverified.')
    (a.output/'pins.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    draw=ImageDraw.Draw(image)
    for ic in output['ics']:
        for pin in ic['pins']:
            if pin['text_source']=='rotated_local_ocr_library_and_vertical_lead_support':
                x,y=pin['body_contact_xy'];draw.ellipse((x-3,y-3,x+3,y+3),outline='green',width=2);draw.text((x+4,y),pin['number']+':'+pin['pinname'],fill='green')
    image.save(a.output/'vertical_contacts.png');print(json.dumps(events))


if __name__=='__main__':main()
