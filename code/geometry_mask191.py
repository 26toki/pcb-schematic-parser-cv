import json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;B=R.parent
def rectangles(image):
 a=np.asarray(image.convert('RGB')).astype(int);red=((a[:,:,0]>100)&(a[:,:,0]>a[:,:,1]+45)&(a[:,:,0]>a[:,:,2]+45)).astype('uint8')
 h=cv2.morphologyEx(red,cv2.MORPH_OPEN,np.ones((1,80),np.uint8));n,_,stats,_=cv2.connectedComponentsWithStats(h,8);lines=[]
 for x,y,w,ht,_ in stats[1:]:
  if w>=100 and ht<=4:lines.append((int(x),int(y+ht//2),int(x+w-1)))
 result=[]
 for x,y,u in lines:
  for a,z,d in lines:
   if z-y<60 or abs(a-x)>3 or abs(d-u)>3:continue
   if not all(red[y:z+1,max(0,e-1):e+2].any(axis=1).mean()>=.98 for e in (x,u)):continue
   contacts=[]
   for edge,sign in [(x,-1),(u,1)]:
    ys=[]
    for yy in range(y+6,z-5):
     xs=[edge+sign*k for k in range(3,9)]
     if min(xs)<0 or max(xs)>=red.shape[1]:continue
     if red[yy,xs].mean()>=.9 and (not ys or yy-ys[-1]>5):ys.append(yy)
    contacts.append(ys)
   # Page borders can also form large rectangles; require dense regular pin rows.
   regular=all(len(rows)>=6 and max(np.diff(rows))<=1.25*min(np.diff(rows)) for rows in contacts)
   if regular:result.append(dict(bbox=[x,y,u,z],outside_red_rows=contacts))
 return result
