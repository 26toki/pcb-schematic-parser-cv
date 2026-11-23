"""Image-derived orthogonal wire graph and observed IC contact groups; no GT inputs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


class Union:
    def __init__(self,n):self.parent=list(range(n))
    def find(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i
    def join(self,a,b):self.parent[self.find(a)]=self.find(b)


def dot_evidence(mask,x,y):
    """A junction adds ink outside both measured wire-stroke bands.

    Fixed 2px corner samples mistake a thick undotted cross for a dot.
    Measure the arms away from the crossing before checking diagonal ink.
    """
    x,y=round(x),round(y)
    if not (12<=x<mask.shape[1]-12 and 12<=y<mask.shape[0]-12):return False
    patch=mask[y-12:y+13,x-12:x+13]
    def band(view):
        candidates=[]
        for columns in (range(0,5),range(20,25)):
            profile=view[:,list(columns)].mean(axis=1)>=.6
            centers=[i for i in range(10,15) if profile[i]]
            if not centers:continue
            mid=min(centers,key=lambda i:abs(i-12));lo=hi=mid
            while lo>0 and profile[lo-1]:lo-=1
            while hi<24 and profile[hi+1]:hi+=1
            if hi-lo+1<=8:candidates.append((hi-lo+1,lo-12,hi-12))
        if not candidates:return None
        _,lo,hi=min(candidates);return lo,hi
    horizontal,vertical=band(patch),band(patch.T)
    if horizontal is None or vertical is None:return False
    h0,h1=horizontal;v0,v1=vertical;samples=[]
    for dx in (v0-3,v0-2,v0-1,v1+1,v1+2,v1+3):
        for dy in (h0-3,h0-2,h0-1,h1+1,h1+2,h1+3):
            if -12<=dx<=12 and -12<=dy<=12:samples.append(bool(patch[12+dy,12+dx]))
    return sum(samples)/max(1,len(samples))>=.35


def connect_segments(segments,mask):
    from wire_join_continuity import existing_ink_join
    uf=Union(len(segments));crossings=[]
    for i,a in enumerate(segments):
        for j in range(i+1,len(segments)):
            b=segments[j]
            if a['direction']==b['direction']:
                if abs(a['axis']-b['axis'])<=1.5 and max(a['start'],b['start'])<=min(a['end'],b['end'])+2 and existing_ink_join(a,b,mask):uf.join(i,j)
                continue
            h,v=(a,b) if a['direction']=='horizontal' else (b,a)
            x,y=v['axis'],h['axis']
            if h['start']-2<=x<=h['end']+2 and v['start']-2<=y<=v['end']+2:
                interior=h['start']+3<x<h['end']-3 and v['start']+3<y<v['end']-3
                dotted=dot_evidence(mask,x,y)
                geometric_join=not interior or dotted
                joined=geometric_join and existing_ink_join(h,v,mask)
                if joined:uf.join(i,j)
                crossings.append(dict(xy=[x,y],joined=joined,kind='missing_local_ink_path' if geometric_join and not joined else 'dot' if dotted else 'four_way_without_dot' if interior else 'endpoint_or_t_junction'))
    return uf,crossings


def line_segments(mask):
    binary=mask.astype(np.uint8)
    segments=[]
    for direction,kernel in [('horizontal',(12,1)),('vertical',(1,12))]:
        cleaned=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,kernel))
        count,labels,stats,centers=cv2.connectedComponentsWithStats(cleaned,8)
        for index in range(1,count):
            x,y,w,h,_=map(int,stats[index]);horizontal=direction=='horizontal'
            if (w if horizontal else h)<12:continue
            # Thick or branched remnants are retained as uncertainty, not curved wires.
            observed_band=None
            if (h if horizontal else w)>10:
                from junction_band162 import axial_band
                observed_band=axial_band(labels[y:y+h,x:x+w]==index,(x,y),direction,mask,dot_evidence)
                if observed_band is None:continue
            axis=observed_band['axis'] if observed_band is not None else float(centers[index][1 if horizontal else 0])
            start=x if horizontal else y;end=x+w-1 if horizontal else y+h-1
            if observed_band is not None:
                start=(x if horizontal else y)+observed_band['start_offset'];end=(x if horizontal else y)+observed_band['end_offset']
            segments.append(dict(id=f'segment_{len(segments):04d}',direction=direction,axis=axis,start=start,end=end))
            if observed_band is not None:segments[-1]['junction_band_evidence']=observed_band
    return segments


def masks_passive_body(component):
    """Observed named component bodies block wires even when value OCR is missing."""
    return bool(component['accepted'] or (component.get('designator') and component.get('geometry_source') in
                 ('observed_rectangle_with_inline_leads','observed_beveled_rectangle_with_inline_leads','observed_alternating_zigzag_with_inline_leads','observed_parallel_plates_with_inline_leads','observed_curved_and_straight_plates_with_inline_leads','observed_colored_coil_arcs_with_inline_leads')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('pins','associations','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--passives',type=Path)
    p.add_argument('--ocr',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    pins=json.loads(a.pins.read_text(encoding='utf-8'))
    associations=json.loads(a.associations.read_text(encoding='utf-8'))['predictions']
    passives=json.loads(a.passives.read_text(encoding='utf-8'))['components'] if a.passives else None
    image=Image.open(pins['image']).convert('RGB');original=np.asarray(image.convert('L'))<160
    mask=original.copy();masked=[]
    def remove(box,reason):
        x,y,u,v=box;x,y=max(0,int(np.floor(x))),max(0,int(np.floor(y)));u,v=min(image.width,int(np.ceil(u))+1),min(image.height,int(np.ceil(v))+1)
        mask[y:v,x:u]=False;masked.append(dict(bbox_xyxy=[x,y,u,v],reason=reason))
    from large_resistor161 import mask_override
    suppressed_masks=[]
    for ic in pins['ics']:
        evidence=mask_override(ic,passives or [])
        if evidence is not None:
            suppressed_masks.append(dict(ic_id=ic['detection_id'],**evidence));continue
        x,y,u,v=ic['body_bbox_xyxy']
        remove([x-3,y-3,u+3,v+3],'observed_ic_body_and_border_with_stroke_margin')
    if passives is None:
        for d in associations:
            if d['type'] in ('r','c') and d['detector_confidence']>=.5:
                remove(d.get('raw_bbox_xyxy',d['body_bbox_xyxy']),'detected_passive_body')
    else:
        for component in passives:
            if masks_passive_body(component):
                x,y,u,v=component.get('wire_mask_bbox_xyxy',component['body_bbox_xyxy'])
                remove([x-1,y-1,u+1,v+1],'observed_named_or_accepted_passive_body_with_stroke_margin')
    from bright_wire_pixels import recover_bright_pixels,encode_runs
    anchors=[]
    for ic in pins['ics']:
        for pin in ic['pins']:
            if pin.get('body_contact_xy') is not None:
                anchors.append(dict(xy=pin['body_contact_xy'],direction='horizontal' if pin['side'] in ('left','right') else 'vertical'))
    for passive in passives or []:
        if passive['accepted']:
            for point in passive['terminal_contacts_xy']:
                for direction in (['horizontal','vertical'] if passive['type']=='l' else [passive['axis']]):
                    anchors.append(dict(xy=point,direction=direction))
    bright,bright_evidence=recover_bright_pixels(np.asarray(image),[r['bbox_xyxy'] for r in masked],anchors)
    original|=bright;mask|=bright
    if a.ocr is not None:
        from ground_bar_mask153 import ground_bar_mask
        excluded_bars,bar_evidence=ground_bar_mask(image,json.loads(a.ocr.read_text(encoding='utf-8'))['items'],[ic['body_bbox_xyxy'] for ic in pins['ics']])
        mask[excluded_bars]=False
        original[excluded_bars]=False
        (a.output/'ground_bar_mask.json').write_text(json.dumps(bar_evidence),encoding='utf-8')
    # Experimental image-only isolation after bright-pixel recovery.
    from geometry_mask191 import rectangles
    missing=[]
    for proposal in rectangles(image):
        box=proposal['bbox']
        if any(max(abs(a-b) for a,b in zip(box,ic['body_bbox_xyxy']))<=5 for ic in pins['ics']):continue
        missing.append(proposal)
        x,y,u,v=box
        remove([x-3,y-3,u+3,v+3],'unrecognized_observed_regular_pin_rectangle')
        original[max(0,y-3):v+4,max(0,x-3):u+4]=False
    (a.output/'geometry_mask191.json').write_text(json.dumps(missing),encoding='utf-8')
    segments=line_segments(mask);uf,crossings=connect_segments(segments,original)
    groups=defaultdict(list);bound=[];unbound=[];contact_conflicts=[]
    for ic in pins['ics']:
        clusters=defaultdict(list)
        for pin in ic['pins']:
            if pin.get('body_contact_xy') is not None:
                clusters[tuple(pin['body_contact_xy'])].append(pin)
        selected=[]
        for point,options in clusters.items():
            if len({p['number'] for p in options})>1:
                exact=[p for p in options if p.get('library_status')=='exact_match']
                accepted=exact if len(exact)==1 else []
                contact_conflicts.append(dict(ic_id=ic['detection_id'],contact_xy=list(point),
                                              candidate_numbers=[p['number'] for p in options],
                                              accepted_numbers=[p['number'] for p in accepted],
                                              reason='conflicting_numbers_at_same_contact; unique_library_supported_pair_or_leave_unresolved'))
                selected.extend(accepted)
            else:selected.extend(options)
        for pin in selected:
            contact=pin.get('body_contact_xy')
            if contact is None:continue
            x,y=contact
            if pin['side'] in ('left','right'):
                x+=-3 if pin['side']=='left' else 3;direction='horizontal';cross,along=y,x
            elif pin['side'] in ('top','bottom'):
                y+=-3 if pin['side']=='top' else 3;direction='vertical';cross,along=x,y
            else:raise ValueError('Unsupported pin side')
            choices=[]
            for i,s in enumerate(segments):
                if s['direction']!=direction:continue
                dy=abs(s['axis']-cross);dx=max(s['start']-along,along-s['end'],0)
                if dy<=3 and dx<=4:choices.append((dy+dx,i))
            ref=dict(kind='ic',ic_id=ic['detection_id'],designator=ic['designator'],number=pin['number'],pinname=pin['pinname'],contact_xy=contact)
            if not choices:
                from contact_pixel_path import bind_short_observed_path
                recovery=bind_short_observed_path(contact,pin['side'],segments,original,
                    [uf.find(i) for i in range(len(segments))],masked,ic['body_bbox_xyxy'])
                if recovery is not None:
                    index,evidence=recovery;choices=[(0,index)];ref['binding_evidence']=evidence
            if not choices:unbound.append(ref);continue
            _,index=min(choices);component=uf.find(index)
            ref['segment_id']=segments[index]['id'];groups[component].append(ref);bound.append(ref)
    passive_contacts=0
    for passive in passives or []:
        if not passive['accepted']:continue
        for terminal,point in enumerate(passive['terminal_contacts_xy'],1):
            passive_contacts+=1;sign=passive.get('terminal_outward_signs',[-1,1])[terminal-1]
            choices=[]
            for i,s in enumerate(segments):
                if s['direction']!=passive['axis'] and passive['type']!='l':continue
                x,y=point
                # Move past a body mask only along an inline wire. A wire that
                # turns at a coil endpoint must be queried at the observed turn.
                if s['direction']==passive['axis']:
                    if passive['axis']=='horizontal':x+=sign*3
                    else:y+=sign*3
                else:
                    # A one-sided coil can turn immediately into an external
                    # wire. Clear the body mask along that outward direction,
                    # rather than leaving the query inside the removed margin.
                    b=passive['body_bbox_xyxy']
                    if s['direction']=='vertical':y+=3 if y>=(b[1]+b[3])/2 else -3
                    else:x+=3 if x>=(b[0]+b[2])/2 else -3
                cross,along=(y,x) if s['direction']=='horizontal' else (x,y)
                da=abs(s['axis']-cross);db=max(s['start']-along,along-s['end'],0)
                if da<=3 and db<=4:choices.append((da+db,i))
            ref=dict(kind='passive',component_id=passive['id'],designator=passive['designator'],type=passive['type'],
                     number=f'terminal_{terminal}',contact_xy=point,internal_terminal_order=True)
            if not choices and passive.get('geometry_source')=='observed_alternating_zigzag_with_inline_leads':
                from contact_pixel_path import bind_passive_lead_turn
                recovery=bind_passive_lead_turn(point,passive['axis'],sign,segments,original,
                    [uf.find(i) for i in range(len(segments))],masked,passive['body_bbox_xyxy'])
                if recovery is not None:
                    index,evidence=recovery;choices=[(0,index)];ref['binding_evidence']=evidence
            if not choices:unbound.append(ref);continue
            _,index=min(choices);component=uf.find(index)
            ref['segment_id']=segments[index]['id'];groups[component].append(ref);bound.append(ref)
    networks=[];draw=ImageDraw.Draw(image);palette=['red','blue','green','purple','orange','brown']
    for component,refs in sorted(groups.items()):
        ids=[i for i in range(len(segments)) if uf.find(i)==component]
        network=dict(id=f'wire_group_{len(networks):03d}',pins=refs,segment_ids=[segments[i]['id'] for i in ids])
        networks.append(network);color=palette[(len(networks)-1)%len(palette)]
        for i in ids:
            s=segments[i];pts=[(s['start'],s['axis']),(s['end'],s['axis'])] if s['direction']=='horizontal' else [(s['axis'],s['start']),(s['axis'],s['end'])]
            draw.line(pts,fill=color,width=3)
    summary=dict(observed_contacts=sum(p['body_contact_xy'] is not None for ic in pins['ics'] for p in ic['pins'])+passive_contacts,
                 bound_contacts=len(bound),unbound_contacts=len(unbound),wire_groups_with_observed_contacts=len(networks),
                 wire_groups_with_ic_contacts=sum(any(p.get('kind','ic')=='ic' for p in n['pins']) for n in networks),
                 passive_contact_candidates=passive_contacts,bound_passive_contacts=sum(p.get('kind')=='passive' for p in bound),
                 multi_pin_groups=sum(len(n['pins'])>1 for n in networks),segments=len(segments),
                 conflicting_contact_locations=len(contact_conflicts),
                 unresolved_four_way_crossings=sum(not c['joined'] for c in crossings))
    report=dict(image=pins['image'],summary=summary,networks=networks,unbound_contacts=unbound,segments=segments,
                crossings=crossings,masked_regions=masked,suppressed_ic_masks=suppressed_masks,
                bright_wire_runs=encode_runs(bright),bright_wire_evidence=bright_evidence,
                contact_conflicts=contact_conflicts,
                provenance=dict(pin_file_sha256=hashlib.sha256(a.pins.read_bytes()).hexdigest()),
                limitations='Image-only local wire candidates. Axis-aligned lines, heuristic dot test; ambiguous four-way crossings kept disconnected. Passive masks may cut real wires or miss symbol strokes. Optional passive terminals use internal order, not official pin numbers. No net-label/global GND/VCC merging or official netlist/scoring.')
    (a.output/'wire_networks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    image.save(a.output/'wire_groups.png')
    Image.fromarray((mask*255).astype(np.uint8)).save(a.output/'masked_wire_pixels.png')
    print(json.dumps(summary))


if __name__=='__main__':main()
