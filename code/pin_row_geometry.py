"""Physical lead rows and literal text confirmation independent of library names."""
import math,re
import numpy as np
from PIL import Image,ImageOps

def observed_side_rows(mask,box,side):
    edge=round(box[0] if side=='left' else box[2]);sign=-1 if side=='left' else 1
    xs=[edge+sign*k for k in range(3,13)]
    if min(xs)<0 or max(xs)>=mask.shape[1]:return []
    top=max(0,math.ceil(box[1])+3);bottom=min(mask.shape[0],math.floor(box[3])-2)
    if bottom<=top:return []
    profile=mask[top:bottom,xs].mean(axis=1)>=.8
    edges=np.diff(np.r_[False,profile,False].astype(np.int8));rows=[]
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if b-a<=6:rows.append(float(top+(a+b-1)/2))
    return rows

def belongs_to_row(glyph_y,row,physical_rows):
    return not physical_rows or min(abs(glyph_y-r) for r in physical_rows)+1>=abs(glyph_y-row)

def confirm_literal_name(image,item,engine,number,library_names):
    text=item['text'].strip();b=item['bbox_xyxy']
    if not library_names or not number.isdigit() or item.get('confidence',0)<.97:return None
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_+./()\-]{0,31}',text):return None
    if b[2]-b[0]>180 or b[3]-b[1]>45 or b[3]-b[1]<6:return None
    crop=image.crop((max(0,math.floor(b[0])-1),max(0,math.floor(b[1])-1),min(image.width,math.ceil(b[2])+1),min(image.height,math.ceil(b[3])+1)))
    crop=ImageOps.expand(crop,border=3,fill='white');confidences=[]
    for scale in [3,4]:
        try:result,_=engine(np.asarray(crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.LANCZOS)),use_det=False,use_cls=False,use_rec=True)
        except Exception:return None
        if not result or len(result)!=1:return None
        read,confidence=result[0][:2]
        if read.strip()!=text or confidence<.97:return None
        confidences.append(float(confidence))
    return dict(observed_name=text,source='literal_name_reconfirmed_at_two_scales_despite_library_disagreement',
        original_confidence=item['confidence'],repeat_confidences=confidences,library_names=library_names)
