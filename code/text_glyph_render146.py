import numpy as np
import cv2
from PIL import Image,ImageOps
def render(crop,target):
    a=np.asarray(crop.convert('RGB')).copy();a[a.max(axis=2).astype(int)-a.min(axis=2).astype(int)>40]=255;dark=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)<160;n,lab,stats,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8),8);keep=np.zeros_like(dark)
    for i,s in enumerate(stats[1:],1):
        if s[4]>=2:keep|=lab==i
    yy,xx=np.where(keep)
    if not len(xx):return None
    chip=Image.fromarray(a).crop((int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1));return ImageOps.expand(chip.resize((max(1,round(chip.width*target/chip.height)),target),Image.Resampling.LANCZOS),border=16,fill='white')
