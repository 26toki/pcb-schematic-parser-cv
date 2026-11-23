"""Separate merged labels using a dominant image gap, not a guessed digit split."""
import math,re
import numpy as np
from PIL import Image,ImageOps
from text_association import associate,box_distance,center
VALUE=re.compile(r'(?:\d+(?:\.\d+)?[kKMm](?:[ΩΩ])?|\d+(?:\.\d+)?[ΩΩ]|\d+[RrKkMm]\d+|\d+[Rr])')
MERGED=re.compile(r'R\d{2,8}(?:[kKMm](?:[ΩΩ])?|[ΩΩ])')

def split_merged_label(image,item,engine):
    if item.get('confidence',0)<.9 or not MERGED.fullmatch(''.join(item['text'].split())):return None
    b=item['bbox_xyxy'];l=max(0,math.floor(b[0])-4);t=max(0,math.floor(b[1])-4)
    r=min(image.width,math.ceil(b[2])+5);bt=min(image.height,math.ceil(b[3])+5)
    if not 20<=r-l<=200 or not 8<=bt-t<=90:return None
    a=np.asarray(image.crop((l,t,r,bt)).convert('RGB')).astype(np.int16)
    blue=(a[:,:,2]-a[:,:,0]>25)&(a[:,:,2]-a[:,:,1]>25)&(a[:,:,2]>60)
    neutral=(np.ptp(a,axis=2)<=35)&(a.mean(axis=2)<160)
    mask=blue if blue.sum()>=15 else neutral
    edges=np.diff(np.pad(mask.any(axis=0).astype(np.int8),(1,1)))
    runs=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    gaps=sorted([(int(runs[i+1][0]-runs[i][1]),int((runs[i+1][0]+runs[i][1])/2)) for i in range(len(runs)-1)],reverse=True)
    if not gaps or gaps[0][0]<3 or (len(gaps)>1 and gaps[0][0]<1.4*gaps[1][0]):return None
    cut=gaps[0][1];panes=[];heights=[];votes=[]
    for start,end in [(0,cut),(cut,mask.shape[1])]:
        yy,xx=np.where(mask[:,start:end])
        if not len(xx):return None
        x0=int(xx.min())+start;x1=int(xx.max())+start+1;y0=int(yy.min());y1=int(yy.max())+1
        if y1-y0<6 or x1-x0<5:return None
        heights.append(y1-y0)
        pane=Image.fromarray(np.where(mask[y0:y1,x0:x1,None],a[y0:y1,x0:x1],255).astype(np.uint8))
        panes.append((ImageOps.expand(pane,border=2,fill='white'),[l+x0,t+y0,l+x1,t+y1]))
    if max(heights)>1.35*min(heights):return None
    for part,(patch,box) in enumerate(panes):
        agreement=[];confidence=[]
        for scale in [3,4]:
            # The image gap already isolated one word. Avoid a second text
            # detector splitting the reference into unrelated R and digits.
            try:raw,_=engine(np.asarray(patch.resize((patch.width*scale,patch.height*scale),Image.Resampling.LANCZOS)),use_det=False,use_cls=False)
            except Exception:return None
            tokens=[(text.strip(),float(conf)) for text,conf in raw or [] if conf>=.8 and re.search(r'[A-Za-z0-9ΩΩ]',text)]
            if len(tokens)!=1 or tokens[0][1]<.95:return None
            text,conf=tokens[0]
            if part==0 and not re.fullmatch(r'R\d+',text):return None
            if part==1 and not VALUE.fullmatch(text):return None
            agreement.append(text);confidence.append(conf)
        if agreement[0]!=agreement[1]:return None
        votes.append(dict(id=f"split_{item['id']}_{part}",text=agreement[0],confidence=min(confidence),bbox_xyxy=box,
            source='two_scale_OCR_of_observed_word_gap',parent_ocr_id=item['id']))
    if ''.join(t['text'] for t in votes)!=''.join(item['text'].split()):return None
    return dict(tokens=votes,gap_px=gaps[0][0],crop_xyxy=[l,t,r,bt],source='observed_dominant_gap_and_two_scale_reference_value_agreement')

def label_zigzag_geometries(image,geometries,items,engine):
    from local_zigzag_text import read_side_label
    result=[];attempts=0;local_attempts=0;cache={}
    for c in geometries:
        body=c['body_bbox_xyxy'];extra=[];evidence=[]
        for item in items:
            if attempts>=6:break
            if box_distance(body,item['bbox_xyxy'])>60 or not MERGED.fullmatch(''.join(item['text'].split())):continue
            if item['id'] not in cache:
                attempts+=1;cache[item['id']]=split_merged_label(image,item,engine)
            split=cache[item['id']]
            if split is not None:extra.extend(split['tokens']);evidence.append(split)
        detection=dict(id='zigzag',class_name='r',body_bbox_xyxy=body)
        labels=associate([detection],[*items,*extra],{},70,20)[0][0]
        valid=bool(labels.get('designator') and labels.get('value') and VALUE.fullmatch(labels['value']))
        valid=valid and all(labels['evidence'].get(f,{}).get('ocr_confidence',0)>=.95 for f in ['designator','value'])
        tokens={t['id']:t for t in [*items,*extra]}
        valid=valid and all(box_distance(body,tokens[labels['evidence'][f]['ocr_id']]['bbox_xyxy'])<=45 for f in ['designator','value'])
        if not valid:
            if engine is None or local_attempts>=32:continue
            local_attempts+=1;labels=read_side_label(image,c,geometries,engine)
            if labels is None:continue
        c=dict(c);c['image_text_label']=labels;c['merged_label_evidence']=evidence;result.append(c)
    return result,dict(attempted_merged_labels=attempts,attempted_side_labels=local_attempts)
