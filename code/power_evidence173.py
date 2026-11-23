import math
import cv2,numpy as np
from PIL import Image
def proposals(image):
 a=np.asarray(image.convert('RGB')).astype(int);r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
 red=((r-g>40)&(r-b>40)&(r>60)).astype('uint8');green=(g-r>30)&(g-b>30)
 n,lab,stats,centers=cv2.connectedComponentsWithStats(red,8);out=[]
 for idx,(x,y,w,h,area) in enumerate(stats[1:],1):
  if not(8<=w<=65 and 8<=h<=65):continue
  chip=lab[y:y+h,x:x+w]==idx
  for rot in range(4):
   m=np.rot90(chip,rot);hh,ww=m.shape
   if not(8<=hh<=40 and 12<=ww<=60):continue
   cols=m.sum(0);bar=np.where(cols>=.8*hh)[0]
   if not len(bar) or len(bar)>4 or bar.min()>1 or bar.max()>3:continue
   mid=(hh-1)/2;stem=m[:,bar.max()+1:];ys,xs=np.where(stem)
   if not len(ys) or np.max(np.abs(ys-mid))>3 or np.mean(stem.any(0))<.9:continue
   # Locate the far stem endpoint in original coordinates.
   marker=np.zeros_like(m,dtype='uint8');marker[round(mid),-1]=1
   yy,xx=np.where(np.rot90(marker,-rot));px=int(x)+int(xx[0]);py=int(y)+int(yy[0])
   dx,dy=[(1,0),(0,1),(-1,0),(0,-1)][rot]
   hits=[]
   for d in range(1,13):
    qx,qy=px+dx*d,py+dy*d
    if not(1<=qx<a.shape[1]-1 and 1<=qy<a.shape[0]-1):hits.append(False);continue
    hits.append(bool(green[qy-1:qy+2,qx-1:qx+2].any()))
   if sum(hits)<9:continue
   out.append(dict(body_bbox_xyxy=[int(x),int(y),int(x+w),int(y+h)],rotation=rot,wire_start=[px,py],green_samples=sum(hits)))
 return out

def gate(record,c):
 p=record['proposal'];b=p['body_bbox_xyxy'];dx,dy=[(1,0),(0,1),(-1,0),(0,-1)][p['rotation']]
 tip=p['wire_start'];axis=0 if dx else 1;sign=dx or dy
 bar=(b[axis] if sign>0 else b[axis+2]);cross=(b[1-axis]+b[3-axis])/2
 labels=[]
 for t in record['high_confidence_voltage']:
  q=t['bbox_xyxy'];cx,cy=(q[0]+q[2])/2,(q[1]+q[3])/2;v=[cx,cy]
  if not(-40<=sign*(v[axis]-bar)<=-2 and abs(v[1-axis]-cross)<=20):continue
  if any(z['body_bbox_xyxy'][0]<=cx<=z['body_bbox_xyxy'][2] and z['body_bbox_xyxy'][1]<=cy<=z['body_bbox_xyxy'][3] for z in c['ics']):continue
  labels.append(t)
 if len({t['text'] for t in labels})!=1:return None,'conflicting_or_missing_label'
 pairs=[(a,b) for i,a in enumerate(labels) for b in labels[i+1:] if a['rotation']!=b['rotation'] and max(abs(x-y) for x,y in zip(a['bbox_xyxy'],b['bbox_xyxy']))<=4]
 if not pairs:return None,'no_spatial_two_view_agreement'
 if any(math.dist([(s['body_bbox_xyxy'][0]+s['body_bbox_xyxy'][2])/2,(s['body_bbox_xyxy'][1]+s['body_bbox_xyxy'][3])/2],[(b[0]+b[2])/2,(b[1]+b[3])/2])<25 for s in c['networks']['power_symbols']):return None,'existing_symbol'
 segs=[]
 for s in c['networks']['segments']:
  if s['direction']!=('horizontal' if dx else 'vertical') or abs(s['axis']-tip[1-axis])>2:continue
  if s['start']-3<=tip[axis]<=s['end']+3:segs.append(s)
 groups=[n for n in c['networks']['networks'] if set(n['segment_ids'])&{s['id'] for s in segs}]
 if not groups and len(segs)==1:
  # Observed but contact-free segment: allow symbol export, never invent a connection.
  gid='isolated_voltage_'+segs[0]['id']
  if any(n['id']==gid for n in c['networks']['networks']):return None,'isolated_group_id_collision'
  group=dict(id=gid,pins=[],segment_ids=[segs[0]['id']],isolated_voltage_group=True)
  return (pairs[0][0],segs[0],group),'accepted_isolated_symbol'
 if len(groups)!=1:return None,'no_unique_existing_group'
 owned=[s for s in segs if s['id'] in groups[0]['segment_ids']]
 if not owned:return None,'no_owned_segment'
 return (pairs[0][0],owned[0],groups[0]),'accepted'
