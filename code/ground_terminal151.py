"""Ground recovery terminals require observed stem direction and ink continuity."""
import math
import numpy as np
def choose_ground_terminal(shape,segments,mask):
    vertical=shape['stem_axis']=='vertical';axis=1 if vertical else 0;across=1-axis
    long,mid,short=shape['bars'];origin=long['center'];outward=-1 if short['center'][axis]>origin[axis] else 1;options=[]
    for index,s in enumerate(segments):
        if s['direction']!=shape['stem_axis'] or s['end']-s['start']<8:continue
        if abs(s['axis']-origin[across])>3:continue
        if max(outward*(s[k]-origin[axis]) for k in ('start','end'))<10:continue
        for endpoint in (s['start'],s['end']):
            delta=outward*(endpoint-origin[axis])
            if not -2<=delta<=18:continue
            evidence=[]
            # Some CAD exports leave a small deliberate gap between the ground
            # glyph and its aligned stem. Do not bridge a break in the stem:
            # require ten continuous pixels starting at the segment endpoint.
            start=max(0,math.ceil(delta)) if 0<=delta<=8 else 0
            for step in range(start,max(start+10,math.ceil(delta))+1):
                along=round(origin[axis]+outward*step);cross=round(origin[across]);x,y=(cross,along) if vertical else (along,cross)
                if not (1<=x<mask.shape[1]-1 and 1<=y<mask.shape[0]-1):evidence=[];break
                strip=mask[y,x-1:x+2] if vertical else mask[y-1:y+2,x];evidence.append(bool(strip.any()))
            if not evidence or np.mean(evidence)<.9:continue
            point=[s['axis'],endpoint] if vertical else [endpoint,s['axis']]
            options.append((abs(delta)+abs(s['axis']-origin[across]),index,point))
    if not options:return None
    options.sort(key=lambda t:t[0])
    if len(options)>1 and options[1][0]-options[0][0]<2 and options[1][1]!=options[0][1]:return None
    return options[0]
