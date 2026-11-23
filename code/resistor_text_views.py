"""Re-read a bounded blue text column, never infer missing characters from a ref."""
import math,re
import numpy as np
from PIL import Image

VALUE=re.compile(r'(?:\d+(?:\.\d+)?[kKMm](?:[ΩΩ])?|\d+(?:\.\d+)?[ΩΩ]|\d+[RrKkMm]\d+|\d+[Rr])')

def confidence(label,field):return label.get('evidence',{}).get(field,{}).get('ocr_confidence',0)

def needs_text_views(component,label):
    if component['type']!='r' or component['axis']!='vertical':return False
    b=component['body_bbox_xyxy']
    if not 8<=b[2]-b[0]<=30 or not 15<=b[3]-b[1]<=80:return False
    return label.get('designator') is None or label.get('value') is None or 0<confidence(label,'designator')<.9 or 0<confidence(label,'value')<.9

def recover_text_views(image,component,candidates,label,engine):
    b=component['body_bbox_xyxy'];neighbors=[c['body_bbox_xyxy'] for c in candidates if c['type']=='r' and c['axis']=='vertical' and c['body_bbox_xyxy'][0]>b[2]
        and min(c['body_bbox_xyxy'][3],b[3])-max(c['body_bbox_xyxy'][1],b[1])>.5*(b[3]-b[1])]
    region=[max(0,math.floor(b[2])-6),max(0,math.floor(b[1])-10),min(image.width,math.ceil(min([p[2]+1 for p in neighbors]+[b[2]+65]))),min(image.height,math.ceil(b[3])+12)]
    report=dict(crop_xyxy=region,votes=[],accepted=False)
    if region[2]-region[0]<18:return None,report
    a=np.asarray(image.crop(region).convert('RGB')).astype(np.int16)
    aa=(a[:,:,2]-a[:,:,0]>10)&(a[:,:,2]-a[:,:,1]>10)&(a[:,:,2]>60)
    binary=(a[:,:,2]-a[:,:,0]>35)&(a[:,:,2]-a[:,:,1]>25)&(a[:,:,2]>60)
    if int(aa.sum())<15:return None,report
    views={'antialiased':Image.fromarray(np.where(aa[:,:,None],a,255).astype(np.uint8)),
           'binary':Image.fromarray(np.repeat(np.where(binary[:,:,None],0,255).astype(np.uint8),3,axis=2))}
    refs=[];values=[];soft_values=[]
    for view,patch in views.items():
        for scale in (3,4):
            try:raw,_=engine(np.asarray(patch.resize((patch.width*scale,patch.height*scale),Image.Resampling.LANCZOS)))
            except Exception as exc:
                report['error']=type(exc).__name__;return None,report
            tokens=[dict(text=t.strip(),confidence=float(s)) for _,t,s in raw or []]
            report['votes'].append(dict(view=view,scale=scale,tokens=tokens))
            if view=='antialiased':
                found={t['text'] for t in tokens if re.fullmatch(r'R\d+',t['text']) and t['confidence']>=.95}
                refs.append(next(iter(found)) if len(found)==1 else None)
                soft_values.extend(t['text'] for t in tokens if VALUE.fullmatch(t['text']) and t['confidence']>=.8)
            else:
                found={t['text'] for t in tokens if VALUE.fullmatch(t['text']) and t['confidence']>=.95}
                values.append(next(iter(found)) if len(found)==1 else None)
    if any(v is None for v in refs+values) or len(set(refs))!=1 or len(set(values))!=1:return None,report
    ref,value=refs[0],values[0]
    if any(v!=value for v in soft_values):return None,report
    oldref,oldvalue=label.get('designator'),label.get('value')
    if oldref is not None and oldref!=ref:
        if not (re.fullmatch(r'R\d+',oldref) and ref.startswith(oldref) and len(ref)>len(oldref) and 0<confidence(label,'designator')<.9):return None,report
    if oldvalue is not None and oldvalue!=value and oldref==ref:
        if not 0<confidence(label,'value')<.9:return None,report
    if (oldref,oldvalue)==(ref,value):return None,report
    report.update(accepted=True,designator=ref,value=value,source='two_scale_antialiased_reference_and_two_scale_binary_value',previous_designator=oldref,previous_value=oldvalue)
    return dict(designator=ref,value=value),report
