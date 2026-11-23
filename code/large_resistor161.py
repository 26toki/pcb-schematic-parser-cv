"""Bounded rescue of large observed rectangles using exact OCR consensus."""
import math,re
import numpy as np
from PIL import Image,ImageOps
VALUE=re.compile(r'(?:\d{3,4}|\d+(?:\.\d+)?(?:[kKMm]?[ΩΩ]|[kKM]|[Rr]))')

def choose_value(views):
    """Four fixed direct-recognition views: RGB/gray at 3x/4x."""
    expected={(v,s) for v in ('rgb','gray') for s in (3,4)}
    if len(views)!=4 or {(v['view'],v['scale']) for v in views}!=expected:return None
    answers=[];scores=[]
    for v in views:
        tokens=v['tokens']
        if len(tokens)!=1:return None
        t=tokens[0];word=t['text'].strip()
        if not VALUE.fullmatch(word) or not math.isfinite(t['confidence']) or not .90<=t['confidence']<=1:return None
        answers.append(word);scores.append(t['confidence'])
    if len(set(answers))!=1 or max(scores)<.95:return None
    return answers[0]

def mask_override(ic,passives):
    """A value token misassigned as box model may be owned by a verified passive."""
    if ic.get('designator') or ic.get('pins'):return None
    a=ic['body_bbox_xyxy'];area=max(0,a[2]-a[0])*max(0,a[3]-a[1]);matches=[]
    if not area:return None
    for p in passives:
        e=p.get('large_resistor_recovery',{});refs=e.get('reference_views',[])
        if not p.get('accepted') or len(refs)!=2 or any(t.get('text')!=p.get('designator') or t.get('confidence',0)<.97 for t in refs):continue
        if choose_value(e.get('value_views',[]))!=p.get('value') or p.get('value') is None:continue
        if ic.get('model') is not None and ic['model'] not in e.get('value_context_texts',[]):continue
        b=p['body_bbox_xyxy'];pb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        if pb and inter/area>=.8 and inter/pb>=.8:matches.append(dict(passive_id=p['id'],reason='dual_reference_and_four_value_views_own_duplicate_box'))
    return matches[0] if len(matches)==1 else None

def recover(image,mask,existing,items,engine,folder,rectangle_fn,recognize,excluded=()):
    used={c.get('designator') for c in existing if c.get('designator')};events=[];result=[]
    proposals=[c for c in rectangle_fn(mask,max_short=60,max_long=200) if min(c['body_bbox_xyxy'][2]-c['body_bbox_xyxy'][0],c['body_bbox_xyxy'][3]-c['body_bbox_xyxy'][1])>31]
    proposals=[c for c in proposals if not any(e[0]<=(c['body_bbox_xyxy'][0]+c['body_bbox_xyxy'][2])/2<=e[2] and e[1]<=(c['body_bbox_xyxy'][1]+c['body_bbox_xyxy'][3])/2<=e[3] for e in excluded)]
    # Existing candidates own the region; ambiguous duplicate references are rejected.
    proposals=[c for c in proposals if not any(max(abs(a-b) for a,b in zip(c['body_bbox_xyxy'],p['body_bbox_xyxy']))<=4 for p in existing)]
    from text_association import associate,box_distance
    for index,c in enumerate(proposals[:4]):
        b=c['body_bbox_xyxy'];f=folder/f'large_{index:02d}';region=[max(0,math.floor(b[0]-55)),max(0,math.floor(b[1]-55)),min(image.width,math.ceil(b[2]+56)),min(image.height,math.ceil(b[3]+56))];views=[];event=dict(box=b,status='rejected',local_ocr_views=2)
        for scale in (3,4):
            recognize(engine,image,region,scale,f/f'context_{scale}x')
            import json
            tokens=json.loads((f/f'context_{scale}x/ocr.json').read_text(encoding='utf-8'))['items'];label=associate([dict(id='large',class_name='r',body_bbox_xyxy=b)],tokens,{},70,20)[0][0];ref=label.get('designator');matches=[t for t in tokens if t['text'].strip()==ref and t['confidence']>=.97]
            views.append(dict(tokens=tokens,reference=ref,reference_token=matches[0] if len(matches)==1 else None))
        if any(v['reference_token'] is None for v in views) or views[0]['reference']!=views[1]['reference'] or not re.fullmatch(r'R\d+',views[0]['reference'] or '') or views[0]['reference'] in used:events.append(event);continue
        refs=[v['reference_token']['bbox_xyxy'] for v in views]
        if math.dist([(refs[0][0]+refs[0][2])/2,(refs[0][1]+refs[0][3])/2],[(refs[1][0]+refs[1][2])/2,(refs[1][1]+refs[1][3])/2])>3:events.append(event);continue
        candidates=[]
        for v in views:
            rb=v['reference_token']['bbox_xyxy'];options=[t for t in v['tokens'] if re.fullmatch(r'\d[\d.A-Za-zΩΩμµ]*',t['text'].strip()) and box_distance(b,t['bbox_xyxy'])<=40 and t['bbox_xyxy'][1]>rb[3] and abs((t['bbox_xyxy'][0]+t['bbox_xyxy'][2]-rb[0]-rb[2])/2)<=max(25,b[2]-b[0])]
            candidates.append(options[0] if len(options)==1 else None)
        if any(t is None for t in candidates):events.append(event);continue
        vb=[sum(t['bbox_xyxy'][i] for t in candidates)/2 for i in range(4)]
        if max(abs(a-bb) for a,bb in zip(candidates[0]['bbox_xyxy'],candidates[1]['bbox_xyxy']))>3:events.append(event);continue
        crop=[max(0,math.floor(vb[0])-5),max(0,math.floor(vb[1])-5),min(image.width,math.ceil(vb[2])+6),min(image.height,math.ceil(vb[3])+6)];patch=image.crop(crop);value_views=[]
        for view,base in [('rgb',patch),('gray',patch.convert('L').convert('RGB'))]:
            for scale in (3,4):
                p=ImageOps.expand(base.resize((base.width*scale,base.height*scale),Image.Resampling.LANCZOS),border=16,fill='white');raw,_=engine(np.asarray(p),use_det=False,use_cls=False);tokens=[dict(text=t,confidence=float(s)) for t,s in raw or []];value_views.append(dict(view=view,scale=scale,tokens=tokens))
        value=choose_value(value_views);event.update(value_views=value_views,value_crop=crop,reference=views[0]['reference'],direct_recognition_views=4)
        if value is None:events.append(event);continue
        c=dict(c);ref=views[0]['reference'];c.update(id=f'large_resistor_{index:02d}',designator=ref,value=value,accepted=True,terminal_numbers_are_internal_order=True,label_source='large_rectangle_dual_reference_four_direct_value_views',large_resistor_recovery=dict(reference_views=[dict(text=v['reference'],confidence=v['reference_token']['confidence'],bbox_xyxy=v['reference_token']['bbox_xyxy']) for v in views],value_views=value_views,value_crop=crop,value_context_texts=[t['text'] for t in candidates]))
        result.append(c);used.add(ref);event['status']='accepted';events.append(event)
    return result,events
