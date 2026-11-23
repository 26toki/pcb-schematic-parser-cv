"""A weak ground requires explicit OCR, three shrinking bars, stem and wire."""
import math
import numpy as np
from text_association import center,box_distance

def ground_ink_mask(image):
    """Recognize dark or saturated colored strokes on a light schematic sheet.

    Used only for already-detected, explicitly labelled weak ground geometry;
    this does not replace the global network mask or invent new wire segments.
    """
    rgb=np.asarray(image.convert('RGB'))
    return rgb.min(axis=2)<160

def _rotate_point(x,y,width,height,k):
    return [(x,y),(y,width-1-x),(width-1-x,height-1-y),(height-1-y,x)][k]

def observe_three_bars(mask,box,contact):
    cx,cy=center(box);l=max(0,round(cx)-45);t=max(0,round(cy)-45)
    r=min(mask.shape[1],round(cx)+46);b=min(mask.shape[0],round(cy)+46)
    patch=mask[t:b,l:r];h,w=patch.shape
    if min(h,w)<20:return None
    options=[]
    for k in range(4):
        rotated=np.rot90(patch,k);px,py=_rotate_point(contact[0]-l,contact[1]-t,w,h,k)
        bx,by=_rotate_point(cx-l,cy-t,w,h,k);bars=[]
        for y,row in enumerate(rotated):
            padded=np.pad(row.astype(np.int8),(1,1));changes=np.diff(padded)
            for x,u in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
                if not 3<=u-x<=80 or abs((x+u-1)/2-bx)>6:continue
                if bars and y==bars[-1][1]+1 and abs(x-bars[-1][2])<=2 and abs(u-bars[-1][3])<=2:
                    bars[-1][1]=y
                else:bars.append([y,y,int(x),int(u)])
        for first in bars:
            y0,y1,x0,u0=first;length0=u0-x0;axis=round((x0+u0-1)/2)
            if abs((y0+y1)/2-by)>32 or abs(px-axis)>3 or not -18<=py-y0<=3:continue
            stem_length=max(4,min(16,round(length0*.25)))
            if y0<stem_length+1 or axis<1 or axis>=rotated.shape[1]-1:continue
            stem=rotated[y0-stem_length:y0,max(0,axis-1):axis+2].any(axis=1)
            if stem.mean()<.9:continue
            for second in bars:
                a,c,x2,u2=second;length2=u2-x2
                gap_min=max(2,round(length0*.05));gap_max=max(6,min(20,round(length0*.4)))
                if not gap_min<=a-y1<=gap_max or not length2<=length0-2:continue
                if abs((x2+u2-1)/2-axis)>2:continue
                for third in bars:
                    d,e,x3,u3=third;length3=u3-x3
                    if not gap_min<=d-c<=gap_max or not 3<=length3<=length2-2:continue
                    if abs((x3+u3-1)/2-axis)>2 or length0<length3*1.5:continue
                    if abs((a-y1)-(d-c))>3:continue
                    # Require the observed wire endpoint and longest bar to be connected in pixels.
                    ya=max(0,round(min(py,y0)));yb=min(rotated.shape[0],round(max(py,y0))+1)
                    if yb<=ya or rotated[ya:yb,axis-1:axis+2].any(axis=1).mean()<.9:continue
                    options.append(dict(rotation_quarter_turns=k,bar_lengths=[length0,length2,length3],
                        bar_positions=[(y0+y1)/2,(a+c)/2,(d+e)/2],stem_axis=axis,
                        source='explicit_GND_OCR_three_decreasing_bars_stem_and_existing_wire_endpoint'))
    return options[0] if len(options)==1 else None

def recover_weak_ground(det,items,ics,segments,mask):
    from attach_power_symbols import valid_label,inside_ic
    if det['class_name']!='gnd' or not .25<=det.get('confidence',0)<.5:return None
    raw=det.get('raw_bbox_xyxy',det['body_bbox_xyxy']);cx,cy=center(raw)
    if inside_ic([cx,cy],ics):return None
    labels=[t for t in items if t.get('confidence',0)>=.95 and valid_label(t['text'],'gnd')
        and box_distance(raw,t['bbox_xyxy'])<=20 and not inside_ic(center(t['bbox_xyxy']),ics)]
    labels.sort(key=lambda t:math.dist([cx,cy],center(t['bbox_xyxy'])))
    if not labels:return None
    if len(labels)>1 and math.dist([cx,cy],center(labels[1]['bbox_xyxy']))-math.dist([cx,cy],center(labels[0]['bbox_xyxy']))<=5:return None
    options=[]
    # A symbol's own horizontal bar can be the nearest extracted "wire".
    # Select an endpoint only after the pixels identify the bar/stem junction.
    for index,s in enumerate(segments):
        length=s['end']-s['start']
        if length<20:continue
        for end in [s['start'],s['end']]:
            point=[s['axis'],end] if s['direction']=='vertical' else [end,s['axis']]
            if math.dist(point,[cx,cy])>48:continue
            evidence=observe_three_bars(mask,raw,point)
            if evidence is not None:options.append(((math.dist(point,[cx,cy])-.02*min(100,length),index,point),evidence))
    if not options:return None
    if any(math.dist(options[0][0][2],other[0][2])>4 for other in options[1:]):return None
    terminal,evidence=min(options,key=lambda pair:pair[0][0])
    evidence.update(detector_confidence=det['confidence'],ocr_confidence=labels[0]['confidence'])
    return (labels[0],'gnd'),terminal,evidence
