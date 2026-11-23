"""Require observed local ink before joining geometrically close wire segments."""
from collections import deque
import math

def existing_ink_join(a,b,mask):
    if a['direction']==b['direction']:
        low=max(a['start'],b['start']);high=min(a['end'],b['end'])
        u=(low+high)/2 if low<=high else min(a['end'],max(a['start'],b['start']))
        v=min(b['end'],max(b['start'],u))
        pa=[u,a['axis']] if a['direction']=='horizontal' else [a['axis'],u]
        pb=[v,b['axis']] if b['direction']=='horizontal' else [b['axis'],v]
    else:
        h,v=(a,b) if a['direction']=='horizontal' else (b,a)
        x,y=v['axis'],h['axis']
        pa=[min(h['end'],max(h['start'],x)),y]
        pb=[x,min(v['end'],max(v['start'],y))]
        a,b=h,v
    left=max(0,math.floor(min(pa[0],pb[0]))-2);right=min(mask.shape[1]-1,math.ceil(max(pa[0],pb[0]))+2)
    top=max(0,math.floor(min(pa[1],pb[1]))-2);bottom=min(mask.shape[0]-1,math.ceil(max(pa[1],pb[1]))+2)
    def support(point,segment):
        x,y=map(round,point);horizontal=segment['direction']=='horizontal'
        return {(xx,yy) for yy in range(max(top,y-1),min(bottom,y+1)+1) for xx in range(max(left,x-1),min(right,x+1)+1)
            if mask[yy,xx] and segment['start']<= (xx if horizontal else yy) <=segment['end']
            and abs((yy if horizontal else xx)-segment['axis'])<=1.5}
    start=support(pa,a);goal=support(pb,b)
    if not start or not goal:return False
    seen=set(start);queue=deque(start)
    while queue:
        x,y=queue.popleft()
        if (x,y) in goal:return True
        for dx,dy in [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]:
            xx,yy=x+dx,y+dy
            if left<=xx<=right and top<=yy<=bottom and (xx,yy) not in seen and mask[yy,xx]:
                seen.add((xx,yy));queue.append((xx,yy))
    return False
