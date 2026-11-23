from functools import lru_cache
from PIL import Image
import numpy as np
@lru_cache(maxsize=32)
def green(path):
 a=np.asarray(Image.open(path).convert('RGB')).astype('int16')
 return (a[:,:,1]>a[:,:,0]+30)&(a[:,:,1]>a[:,:,2]+30)
def mark(items,path):
 mask=green(str(path));height,width=mask.shape;result=[]
 for old in items:
  t=dict(old);x,y,u,v=t['bbox_xyxy'];lo=max(0,int(x));hi=min(width,int(u)+1)
  if hi-lo>=10 and t['confidence']>=.95:
   for yy in range(max(0,int(v)-4),min(height,int(v)+9)):
    if mask[max(0,yy-1):min(height,yy+2),lo:hi].any(axis=0).mean()>=.9:
     t['observed_green_wire_label']=dict(row=yy,coverage_threshold=.9);break
  result.append(t)
 return result
