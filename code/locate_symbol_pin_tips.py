"""Image ground-bar geometry, training power offsets and ambiguous IC tip intervals. No GT."""
import argparse
from collections import Counter
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


def ground_bars(mask,symbol,segment):
    vertical=segment['direction']=='vertical';along=1 if vertical else 0;across=1-along
    contact=symbol['contact_xy'];b=symbol['body_bbox_xyxy'];center=[(b[0]+b[2])/2,(b[1]+b[3])/2]
    sign=1 if center[along]>=contact[along] else -1
    x,y=map(round,contact);left=max(0,x-55);top=max(0,y-55)
    patch=mask[top:min(mask.shape[0],y+56),left:min(mask.shape[1],x+56)].astype(np.uint8)
    # Odd kernel keeps the opening anchor centered; an even kernel shifted centers by 1px.
    kernel=(7,1) if vertical else (1,7)
    opened=cv2.morphologyEx(patch,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,kernel))
    count,_,stats,centers=cv2.connectedComponentsWithStats(opened,8);bars=[]
    for index in range(1,count):
        px,py,w,h,_=stats[index];length=w if vertical else h;thickness=h if vertical else w
        if not 6<=length<=90 or thickness>4:continue
        mid=[float(centers[index][0])+left,float(centers[index][1])+top]
        if abs(mid[across]-contact[across])>5:continue
        bars.append(dict(mid=mid,length=int(length)))
    options=[]
    for triple in itertools.combinations(bars,3):
        ordered=sorted(triple,key=lambda bar:sign*bar['mid'][along]);a,b,c=ordered
        spacing=[sign*(b['mid'][along]-a['mid'][along]),sign*(c['mid'][along]-b['mid'][along])]
        if not all(3<=distance<=18 for distance in spacing):continue
        if max(spacing)/min(spacing)>1.5:continue
        if not a['length']>b['length']>c['length'] or c['length']>.75*a['length']:continue
        if b['length']<.35*a['length'] or c['length']<.35*b['length']:continue
        if abs(a['mid'][along]-contact[along])>18:continue
        if sign*(a['mid'][along]-contact[along]) < -8:continue
        score=abs(a['mid'][along]-contact[along])+sum(abs(bar['mid'][across]-contact[across]) for bar in ordered)
        options.append((score,ordered))
    if not options:return None
    _,bars=min(options,key=lambda option:option[0])
    return dict(position_image_xy=bars[-1]['mid'],source='three_descending_ground_bars_shortest_bar_center',
                observed_bars=bars,verified=False)


def voltage_offset(mask,symbol,segment,priors):
    b=symbol['body_bbox_xyxy'];center=[(b[0]+b[2])/2,(b[1]+b[3])/2]
    options=[];vertical=segment['direction']=='vertical'
    for offset in priors['power_offsets'].get('v',[]):
        point=[center[0]+offset['dx'],center[1]-offset['dy_gt']]
        along,across=(1,0) if vertical else (0,1)
        if (offset['dx']==0)!=vertical:continue
        if abs(point[across]-segment['axis'])>5:continue
        if not segment['start']<=point[along]<=segment['end']:continue
        point[across]=segment['axis'];x,y=map(round,point)
        if not (0<=x<mask.shape[1] and 0<=y<mask.shape[0]):continue
        if not mask[max(0,y-1):y+2,max(0,x-1):x+2].any():continue
        distance=math.dist(point,symbol['contact_xy'])
        if distance>35:continue
        options.append((distance,point,offset))
    if len(options)!=1:return None
    _,point,offset=options[0]
    return dict(position_image_xy=point,source='training_voltage_offset_supported_by_attached_segment',
                offset=offset,verified=False)


def color_pin_tip(rgb,pin):
    """Resolve red symbol leads meeting green wires; require sustained evidence on both sides.

    Work only on the observed lead axis. Blue labels, monochrome lines and isolated
    colored junction dots cannot supply the required contiguous red/green runs.
    """
    if rgb is None or pin.get('body_contact_xy') is None:return None
    x,y=map(round,pin['body_contact_xy']);vertical=pin['side'] in ('top','bottom')
    sign=-1 if pin['side'] in ('left','top') else 1
    labels=[]
    for gap in range(1,81):
        xx=x if vertical else x+sign*gap;yy=y+sign*gap if vertical else y
        if not (0<=xx<rgb.shape[1] and 0<=yy<rgb.shape[0]):break
        strip=(rgb[yy,max(0,xx-1):xx+2] if vertical else rgb[max(0,yy-1):yy+2,xx]).astype(float)
        r,g,b=strip.T
        red=(r-g>35)&(r-b>35)&(r>70)
        green=(g-r>30)&(g-b>30)&(g>60)
        labels.append('r' if red.any() and not green.any() else 'g' if green.any() and not red.any() else '?')
    # At most two antialiased pixels may separate the two contiguous color runs.
    # A near-body red run is essential: do not jump across a wire to nearby text.
    if len(labels)<10 or labels[:6].count('r')<4:return None
    for index in range(6,len(labels)-3):
        if labels[index:index+4]!=['g']*4:continue
        end=index-1
        while end>=0 and labels[end]=='?' and index-end<=3:end-=1
        if end<5 or labels[end-5:end+1]!=['r']*6:return None
        if any(label=='g' for label in labels[:end]):return None
        gap=index+1
        return dict(status='observed_red_pin_green_wire_boundary',
                    position_image_xy=[x if vertical else x+sign*gap,y+sign*gap if vertical else y],
                    gap_px=gap,source='sustained_axis_color_transition',
                    evidence=dict(red_run_min=6,green_run_min=4,max_antialias_gap=2))
    return None


def color_pin_corner_tip(rgb,pin):
    """Observe a red lead ending at a green wire which immediately turns.

    Require six contiguous red lead pixels, a near-body red run, and four
    contiguous green pixels on a perpendicular branch touching that endpoint.
    A lone colored dot or a nearby parallel wire cannot provide this evidence.
    """
    if rgb is None or pin.get('body_contact_xy') is None:return None
    original_x,original_y=map(round,pin['body_contact_xy'])
    left=max(0,original_x-84);top=max(0,original_y-84)
    a=np.asarray(rgb)[top:original_y+85,left:original_x+85].astype(np.int16)
    r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
    red=(r-g>35)&(r-b>35)&(r>70);green=(g-r>30)&(g-b>30)&(g>60)
    x,y=original_x-left,original_y-top;vertical=pin['side'] in ('top','bottom')
    sign=-1 if pin['side'] in ('left','top') else 1
    labels=[];points=[]
    for gap in range(1,81):
        xx,yy=(x,y+sign*gap) if vertical else (x+sign*gap,y)
        if not(2<=xx<a.shape[1]-2 and 2<=yy<a.shape[0]-2):break
        rs=red[yy,xx-1:xx+2] if vertical else red[yy-1:yy+2,xx]
        gs=green[yy,xx-1:xx+2] if vertical else green[yy-1:yy+2,xx]
        labels.append('r' if rs.any() and not gs.any() else 'g' if gs.any() and not rs.any() else '?')
        points.append((xx,yy))
    if len(labels)<10 or labels[:6].count('r')<4:return None
    for index in range(6,len(labels)):
        if labels[index]!='g':continue
        end=index-1
        while end>=0 and labels[end]=='?' and index-end<=3:end-=1
        if end<5 or labels[end-5:end+1]!=['r']*6:return None
        if 'g' in labels[:end]:return None
        xx,yy=points[index];branches=[]
        for direction in [-1,1]:
            evidence=[]
            for offset in range(1,7):
                bx,by=(xx+direction*offset,yy) if vertical else (xx,yy+direction*offset)
                if not(1<=bx<a.shape[1]-1 and 1<=by<a.shape[0]-1):break
                strip=green[by-1:by+2,bx] if vertical else green[by,bx-1:bx+2]
                evidence.append(bool(strip.any()))
            if len(evidence)>=4 and all(evidence[:4]):branches.append(direction)
        if not branches:continue
        return dict(status='observed_red_pin_green_wire_boundary',position_image_xy=[xx+left,yy+top],
                    gap_px=index+1,source='red_lead_end_with_contiguous_perpendicular_green_branch',
                    evidence=dict(red_run_min=6,perpendicular_green_run_min=4,branch_directions=branches))
    return None


def ic_interval(mask,ic,pin,priors,rgb=None):
    if pin.get('body_contact_xy') is None:return dict(status='no_observed_contact')
    from color_boundary158 import robust_tip
    colored=color_pin_tip(rgb,pin) or color_pin_corner_tip(rgb,pin) or robust_tip(rgb,pin)
    if colored is not None:return colored
    x,y=pin['body_contact_xy'];vertical=pin['side'] in ('top','bottom')
    sign=-1 if pin['side'] in ('left','top') else 1
    points=[]
    for item in priors['ic_gap_distribution']:
        gap=item['gap_px']
        if not 4<=gap<=40:continue
        xx=round(x) if vertical else round(x+sign*gap);yy=round(y+sign*gap) if vertical else round(y)
        if not (0<=xx<mask.shape[1] and 0<=yy<mask.shape[0]):continue
        samples=[round((y if vertical else x)+sign*k) for k in range(3,gap+1)]
        if not all(0<=coordinate<(mask.shape[0] if vertical else mask.shape[1]) for coordinate in samples):continue
        coverage=mask[samples,max(0,xx-1):xx+2].any(axis=1).mean() if vertical else mask[max(0,yy-1):yy+2,samples].any(axis=0).mean()
        if coverage<.9:continue
        points.append(dict(position_image_xy=[xx,yy],gap_px=gap,training_count=item['count']))
    if not points:return dict(status='no_supported_training_gap')
    return dict(status='ambiguous_pin_wire_boundary',position_image_xy=None,
                supported_positions=points,position_range_image_xy=[min(p['position_image_xy'][0] for p in points),min(p['position_image_xy'][1] for p in points),
                                                                    max(p['position_image_xy'][0] for p in points),max(p['position_image_xy'][1] for p in points)],
                note='Continuous pixels support every listed gap. Frequency is not confidence; no unique tip chosen.')


def calibrate_peer_pin_lengths(rgb,ic):
    """Use repeated lead geometry of this symbol, never another case or GT.

    At least two distinct contacts must expose consistent red/green boundaries.
    The inferred endpoint remains a hypothesis, separately tagged from direct
    color observation. Heterogeneous lengths or broken red leads are rejected.
    """
    output=copy.deepcopy(ic);a=np.asarray(rgb).astype(np.int16)
    red=(a[:,:,0]-a[:,:,1]>35)&(a[:,:,0]-a[:,:,2]>35)&(a[:,:,0]>70)
    anchors=[];seen=set()
    for p in output['pins']:
        t=p.get('tip_geometry',{});contact=p.get('body_contact_xy')
        if t.get('status')!='observed_red_pin_green_wire_boundary' or contact is None:continue
        key=tuple(contact)
        if key in seen:continue
        seen.add(key);anchors.append(p)
    for p in output['pins']:
        old=p.get('tip_geometry',{});contact=p.get('body_contact_xy')
        if old.get('status')!='ambiguous_pin_wire_boundary' or contact is None:continue
        vertical=p['side'] in ('top','bottom');axis_sides=('top','bottom') if vertical else ('left','right')
        same=[q for q in anchors if q['side']==p['side']]
        peers=same if len(same)>=2 else [q for q in anchors if q['side'] in axis_sides]
        if len(peers)<2:continue
        gaps=[q['tip_geometry']['gap_px'] for q in peers]
        if max(gaps)-min(gaps)>3:continue
        gap=float(np.median(gaps))
        if not 6<=gap<=80:continue
        x,y=contact;sign=-1 if p['side'] in ('left','top') else 1
        samples=[]
        for k in range(3,max(4,round(gap)-1)):
            xx,yy=(round(x),round(y+sign*k)) if vertical else (round(x+sign*k),round(y))
            if not(1<=xx<red.shape[1]-1 and 1<=yy<red.shape[0]-1):samples=[];break
            samples.append(bool(red[yy,max(0,xx-1):xx+2].any() if vertical else red[max(0,yy-1):yy+2,xx].any()))
        if len(samples)<4 or sum(samples)/len(samples)<.9:continue
        point=[float(x if vertical else x+sign*gap),float(y+sign*gap if vertical else y)]
        if not(0<=point[0]<red.shape[1] and 0<=point[1]<red.shape[0]):continue
        p['tip_geometry']=dict(status='same_symbol_consistent_peer_length',position_image_xy=point,gap_px=gap,
            source='distinct_same_symbol_color_boundaries_and_target_red_lead',
            peer_numbers=[q['number'] for q in peers],peer_gaps_px=gaps,red_coverage=sum(samples)/len(samples),
            scope='same_side' if len(same)>=2 else 'same_axis_opposite_side_allowed',verified=False)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pins','networks','priors','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();ics=json.loads(a.pins.read_text(encoding='utf-8'))
    network=json.loads(a.networks.read_text(encoding='utf-8'));priors=json.loads(a.priors.read_text(encoding='utf-8'))
    image=Image.open(network['image']).convert('RGB');rgb=np.asarray(image);mask=np.asarray(image.convert('L'))<160
    segments={s['id']:s for s in network['segments']};symbols=[];ic_results=[];statuses=Counter()
    for symbol in network['power_symbols']:
        result=copy.deepcopy(symbol);segment=segments[symbol['segment_id']]
        evidence=ground_bars(mask,symbol,segment) if symbol['type']=='gnd' else None
        tip=voltage_offset(mask,symbol,segment,priors) if symbol['type']=='v' else None
        result['symbol_geometry_evidence']=evidence
        result['pin_tip_candidate']=tip
        result['pin_tip_status']=('ground_boundary_convention_unresolved' if symbol['type']=='gnd' else
                                 'geometry_hypothesis' if tip else 'unresolved_symbol_geometry')
        statuses[result['pin_tip_status']]+=1;symbols.append(result)
    for ic in ics['ics']:
        result=copy.deepcopy(ic)
        for pin in result['pins']:pin['tip_geometry']=ic_interval(mask,ic,pin,priors,rgb)
        ic_results.append(calibrate_peer_pin_lengths(rgb,result))
    summary=dict(power_symbols=len(symbols),power_tip_hypotheses=sum(s['pin_tip_candidate'] is not None for s in symbols),
                 ground_tip_hypotheses=sum(s['type']=='gnd' and s['pin_tip_candidate'] is not None for s in symbols),
                 ground_bar_evidence=sum(s['symbol_geometry_evidence'] is not None for s in symbols),
                 voltage_tip_hypotheses=sum(s['type']=='v' and s['pin_tip_candidate'] is not None for s in symbols),
                 ambiguous_ic_pins=sum(pin['tip_geometry']['status']=='ambiguous_pin_wire_boundary' for ic in ic_results for pin in ic['pins']),
                 color_resolved_ic_pins=sum(pin['tip_geometry']['status']=='observed_red_pin_green_wire_boundary' for ic in ic_results for pin in ic['pins']),
                 power_statuses=dict(statuses))
    result=dict(image=network['image'],summary=summary,power_symbols=symbols,ics=ic_results,
                provenance={key+'_sha256':hashlib.sha256(path.read_bytes()).hexdigest() for key,path in
                            [('pins',a.pins),('networks',a.networks),('priors',a.priors)]},
                limitations='Ground and voltage retain existing hypotheses. IC color transitions require sustained red lead and green wire evidence; other IC boundaries remain ambiguous. No physical graph changes, no GT or API.')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'symbol_pin_tips.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    draw=ImageDraw.Draw(image)
    for symbol in symbols:
        if symbol['pin_tip_candidate']:
            x,y=symbol['pin_tip_candidate']['position_image_xy'];draw.ellipse((x-4,y-4,x+4,y+4),outline='green',width=2)
    for ic in ic_results:
        for pin in ic['pins']:
            interval=pin['tip_geometry'].get('position_range_image_xy')
            if interval:draw.line(tuple(interval),fill='orange',width=3)
            tip=pin['tip_geometry'].get('position_image_xy')
            if tip:
                x,y=tip;draw.ellipse((x-4,y-4,x+4,y+4),outline='green',width=2)
    image.save(a.output/'symbol_pin_candidates.png');print(json.dumps(summary))


if __name__=='__main__':main()
