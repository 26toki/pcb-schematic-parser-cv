"""Antialiased neutral-text rendering; no name or number substitution."""
import numpy as np
from PIL import Image,ImageOps

def render(crop,height):
    a=np.asarray(crop.convert('RGB')).astype(float)
    neutral=a.max(2)-a.min(2)<40
    ink=neutral&(a.mean(2)<160)
    yy,xx=np.where(ink)
    if len(xx)<4:return None
    gray=np.where(neutral,a@np.array([.299,.587,.114]),255).clip(0,255).astype('uint8')
    chip=Image.fromarray(gray).crop((int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)).convert('RGB')
    return ImageOps.expand(chip.resize((max(1,round(chip.width*height/chip.height)),height),Image.Resampling.LANCZOS),border=16,fill='white')

def accept(name,views):
    return len(views)==2 and all(len(v)==1 and v[0]['text']==name and .95<=v[0]['confidence']<=1 for v in views)
