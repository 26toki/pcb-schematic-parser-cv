"""Bind an IC contact across a short glyph only when existing ink connects it.

No wire, contact coordinate, mask or global joining threshold is changed.
"""
from collections import deque
import math

def _path(mask,contact,target,direction,segment_span,target_direction=None):
    horizontal=direction=='horizontal'
    lo=math.floor(min(contact[0],target[0]));hi=math.ceil(max(contact[0],target[0]))
    top=math.floor(min(contact[1],target[1]));bottom=math.ceil(max(contact[1],target[1]))
    if horizontal:top=round(contact[1])-4;bottom=round(contact[1])+4
    else:lo=round(contact[0])-4;hi=round(contact[0])+4
    lo=max(0,lo);top=max(0,top);hi=min(mask.shape[1]-1,hi);bottom=min(mask.shape[0]-1,bottom)
    if lo>hi or top>bottom:return None
    goal_horizontal=(target_direction or direction)=='horizontal'
    def near(point,start=False):
        x,y=map(round,point)
        return {(xx,yy) for yy in range(y-1,y+2) for xx in range(x-1,x+2)
                if lo<=xx<=hi and top<=yy<=bottom and mask[yy,xx]
                and ((xx if horizontal else yy)==(x if horizontal else y) if start else
                     segment_span[0]<= (xx if goal_horizontal else yy) <=segment_span[1])}
    starts=near(contact,True);goals=near(target)
    if not starts or not goals:return None
    seen=set(starts);queue=deque((p,0) for p in starts)
    while queue:
        (x,y),distance=queue.popleft()
        if (x,y) in goals:return distance
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            p=(x+dx,y+dy)
            if lo<=p[0]<=hi and top<=p[1]<=bottom and p not in seen and mask[p[1],p[0]]:
                seen.add(p);queue.append((p,distance+1))
    return None

def bind_short_observed_path(contact,side,segments,mask,roots,masked_regions,owner_body):
    direction='horizontal' if side in ('left','right') else 'vertical';sign=-1 if side in ('left','top') else 1
    cross_index=1 if direction=='horizontal' else 0;along_index=1-cross_index
    owner_mask=[owner_body[0]-3,owner_body[1]-3,owner_body[2]+4,owner_body[3]+4]
    options=[]
    for index,s in enumerate(segments):
        if s['direction']!=direction or abs(s['axis']-contact[cross_index])>3:continue
        along=min(s['end'],max(s['start'],contact[along_index]))
        distance=sign*(along-contact[along_index])
        if not 3<distance<=18:continue
        target=[along,s['axis']] if direction=='horizontal' else [s['axis'],along]
        corridor=[min(contact[0],target[0]),min(contact[1],target[1]),max(contact[0],target[0]),max(contact[1],target[1])]
        blocked=False
        for region in masked_regions:
            b=region['bbox_xyxy']
            if region['reason'].startswith('observed_ic_body') and max(abs(a-c) for a,c in zip(b,owner_mask))<=2:continue
            if max(corridor[0]-1,b[0])<=min(corridor[2]+1,b[2]) and max(corridor[1]-1,b[1])<=min(corridor[3]+1,b[3]):
                blocked=True;break
        if blocked:continue
        steps=_path(mask,contact,target,direction,(s['start'],s['end']))
        if steps is None:continue
        options.append((distance+abs(s['axis']-contact[cross_index]),index,steps,target))
    if not options or len({roots[o[1]] for o in options})!=1:return None
    score,index,steps,target=min(options)
    return index,dict(source='short_existing_ink_path_from_IC_contact_to_unique_outward_wire_group',
        path_steps=steps,observed_target_xy=target,maximum_outward_distance=18,corridor_half_width=4)

def bind_passive_lead_turn(contact,axis,sign,segments,mask,roots,masked_regions,owner_body):
    """A new zigzag's short inline lead can turn into a perpendicular wire."""
    along_index=0 if axis=='horizontal' else 1;cross_index=1-along_index
    own_mask=[owner_body[0]-1,owner_body[1]-1,owner_body[2]+2,owner_body[3]+2];options=[]
    for index,s in enumerate(segments):
        if s['direction']==axis:continue
        across=min(s['end'],max(s['start'],contact[cross_index]))
        target=[s['axis'],across] if axis=='horizontal' else [across,s['axis']]
        distance=sign*(target[along_index]-contact[along_index])
        if not 4<distance<=18 or abs(target[cross_index]-contact[cross_index])>4:continue
        corridor=[min(contact[0],target[0])-1,min(contact[1],target[1])-1,max(contact[0],target[0])+1,max(contact[1],target[1])+1]
        blocked=False
        for region in masked_regions:
            b=region['bbox_xyxy']
            if region['reason'].startswith('observed_named_or_accepted_passive') and max(abs(x-y) for x,y in zip(b,own_mask))<=2:continue
            if max(corridor[0],b[0])<=min(corridor[2],b[2]) and max(corridor[1],b[1])<=min(corridor[3],b[3]):blocked=True;break
        if blocked:continue
        steps=_path(mask,contact,target,axis,(s['start'],s['end']),s['direction'])
        if steps is not None:options.append((distance,index,steps,target))
    if not options or len({roots[o[1]] for o in options})!=1:return None
    _,index,steps,target=min(options)
    return index,dict(source='existing_short_inline_lead_turn_into_unique_perpendicular_wire',path_steps=steps,observed_target_xy=target)
