"""Image-only wire geometry, separate from electrical grouping and symbol leads."""
import copy
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image
from trace_ic_wire_networks import dot_evidence


def green_wire_mask(rgb):
    pixels=np.asarray(rgb).astype(np.int16);r,g,b=pixels[...,0],pixels[...,1],pixels[...,2]
    return (g-r>25)&(g-b>25)&(g>60)


def green_segments(mask):
    segments=[]
    for direction,kernel in [('horizontal',(5,1)),('vertical',(1,5))]:
        opened=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,kernel))
        count,_,stats,centers=cv2.connectedComponentsWithStats(opened,8)
        for i in range(1,count):
            x,y,w,h,_=map(int,stats[i]);horizontal=direction=='horizontal'
            length,thickness=(w,h) if horizontal else (h,w)
            if thickness>6 or length<max(6,3*thickness):continue
            segments.append(dict(direction=direction,axis=float(centers[i][1 if horizontal else 0]),
                                 start=float(x if horizontal else y),end=float(x+w-1 if horizontal else y+h-1)))
    return segments


def snap_red_junction_centers(segments,rgb):
    """Extend green endpoints across a measured filled red junction disk only.

    Thin rings, arrows and elongated symbol leads are not junction disks. The
    original physical-net grouping is unchanged; only export geometry changes.
    """
    a=np.asarray(rgb).astype(np.int16)
    red=((a[:,:,0]-a[:,:,1]>35)&(a[:,:,0]-a[:,:,2]>35)&(a[:,:,0]>70)).astype(np.uint8)
    dots=[]
    # A filled junction can touch a thin red component lead. An opening removes
    # those stems while retaining the disk, without filling rings or line gaps.
    # Keep original isolated-disk observations first so their centers do not move.
    opened=cv2.morphologyEx(red,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    for mask in (red,opened):
        n,_,stats,centers=cv2.connectedComponentsWithStats(mask,8)
        for s,center in zip(stats[1:],centers[1:]):
            x,y,w,h,area=map(int,s)
            if 6<=min(w,h)<=22 and max(w,h)<=22 and .75<=w/h<=1.33 and area/(w*h)>=.6:
                if not any(np.linalg.norm(center-old)<3 for old,_ in dots):
                    dots.append((center,(max(w,h)-1)/2))
    result=[];snaps=0
    for original in segments:
        s=dict(original);horizontal=s['direction']=='horizontal';along=0 if horizontal else 1;cross=1-along
        for key,sign in [('start',-1),('end',1)]:
            options=[]
            for center,radius in dots:
                extension=sign*(center[along]-s[key]);offset=abs(center[cross]-s['axis'])
                if -.5<=extension<=radius+2 and offset<=2:options.append((offset+extension,center))
            if options:
                _,point=min(options,key=lambda p:p[0]);s[key]=float(point[along]);snaps+=1
        result.append(s)
    return result,snaps


def split_at_junctions(segments,mask):
    """Split at T/dotted crossings, normalize small stroke overshoot at corners."""
    cuts=[[s['start'],s['end']] for s in segments]
    def cut(index,value):
        s=segments[index]
        if abs(value-s['start'])<=3:cuts[index][0]=value
        elif abs(value-s['end'])<=3:cuts[index][1]=value
        else:cuts[index].append(value)
    for i,a in enumerate(segments):
        for j in range(i+1,len(segments)):
            b=segments[j]
            if a['direction']==b['direction']:continue
            hi,vi=(i,j) if a['direction']=='horizontal' else (j,i)
            h,v=segments[hi],segments[vi];x,y=v['axis'],h['axis']
            if not (h['start']-3<=x<=h['end']+3 and v['start']-3<=y<=v['end']+3):continue
            interior=h['start']+3<x<h['end']-3 and v['start']+3<y<v['end']-3
            if interior and not dot_evidence(mask,x,y):continue
            cut(hi,x);cut(vi,y)
    result=[]
    for s,values in zip(segments,cuts):
        values=sorted(set(values))
        for start,end in zip(values,values[1:]):
            if end-start>=3:result.append(dict(s,direction=s['direction'],start=start,end=end))
    return result


def join_text_occlusions(segments,rgb,network):
    """Join short blue-text gaps only inside an already observed physical net.

    Red symbol strokes and masked component bodies forbid a join. Junction
    splitting is performed afterward, so real branches remain separate edges.
    """
    pixels=np.asarray(rgb).astype(np.int16);r,g,b=pixels[...,0],pixels[...,1],pixels[...,2]
    blue=(b-r>25)&(b-g>25)&(b>60);red=(r-g>35)&(r-b>35)&(r>70)
    owners=defaultdict(set)
    for group in network['networks']:
        for sid in group['segment_ids']:owners[sid].add(group['id'])
    def owner(segment):
        votes=defaultdict(float)
        for old in network['segments']:
            if old['direction']!=segment['direction'] or abs(old['axis']-segment['axis'])>3:continue
            overlap=max(0,min(old['end'],segment['end'])-max(old['start'],segment['start']))
            if overlap<3:continue
            for gid in owners[old['id']]:votes[gid]+=overlap
        ranked=sorted(votes.items(),key=lambda x:-x[1])
        if not ranked or len(ranked)>1 and ranked[1][1]>=.8*ranked[0][1]:return None
        return ranked[0][0]
    items=[dict(s,_owner=owner(s)) for s in segments];joins=0;changed=True
    while changed:
        changed=False
        for i,a in enumerate(items):
            if a['_owner'] is None:continue
            for j in range(i+1,len(items)):
                c=items[j]
                if c['_owner']!=a['_owner'] or c['direction']!=a['direction'] or abs(c['axis']-a['axis'])>1:continue
                left,right=sorted((a,c),key=lambda s:s['start']);gap=right['start']-left['end']
                if not 1<gap<=24:continue
                cross=round((a['axis']+c['axis'])/2);lo=round(left['end'])+1;hi=round(right['start'])
                box=[lo,cross-3,hi,cross+3] if a['direction']=='horizontal' else [cross-3,lo,cross+3,hi]
                x,y,u,v=box;x=max(0,x);y=max(0,y);u=min(blue.shape[1],u);v=min(blue.shape[0],v)
                if blue[y:v+1,x:u+1].sum()<2 or red[y:v+1,x:u+1].any():continue
                blocked=False
                for region in network.get('masked_regions',[]):
                    bx,by,bu,bv=region['bbox_xyxy']
                    if max(x,bx)<=min(u,bu) and max(y,by)<=min(v,bv):blocked=True;break
                if blocked:continue
                wa=a['end']-a['start'];wc=c['end']-c['start']
                merged=dict(a,start=left['start'],end=right['end'],axis=(a['axis']*wa+c['axis']*wc)/(wa+wc))
                items[i]=merged;items.pop(j);joins+=1;changed=True;break
            if changed:break
    return [{k:v for k,v in item.items() if k!='_owner'} for item in items],joins


def attach_export_geometry(network,image):
    """Map observed green segments onto existing groups by collinear pixel overlap.

    Keep electrical contacts, groups, and original tracing segments unchanged.
    No component IDs, model names or golden coordinates select wire endpoints.
    Monochrome and unsupported palettes retain existing geometry.
    """
    result=copy.deepcopy(network)
    with Image.open(image) as opened:rgb=np.asarray(opened.convert('RGB'));mask=green_wire_mask(rgb)
    if result.get('bright_wire_runs'):
        from bright_wire_pixels import restore_runs
        restore_runs(mask,result['bright_wire_runs'])
    observed=green_segments(mask)
    if sum(s['end']-s['start'] for s in observed)<100:
        result['export_geometry_summary']=dict(mode='existing_geometry_fallback',reason='insufficient_green_wire_evidence')
        return result
    observed,junction_snaps=snap_red_junction_centers(observed,rgb)
    observed,text_joins=join_text_occlusions(observed,rgb,result)
    # Include red junction dots, but exclude blue text from junction evidence.
    pixels=rgb.astype(np.int16);r,g,b=pixels[...,0],pixels[...,1],pixels[...,2]
    junction_mask=mask|((r-g>35)&(r-b>35)&(r>70))
    split=split_at_junctions(observed,junction_mask)
    owners=defaultdict(list)
    for group in result['networks']:
        for sid in group['segment_ids']:owners[sid].append(group['id'])
        group['export_edges_image_xy']=[]
    groups={g['id']:g for g in result['networks']};unassigned=0
    for s in split:
        votes=defaultdict(float)
        for old in result['segments']:
            if old['direction']!=s['direction'] or abs(old['axis']-s['axis'])>3:continue
            overlap=max(0,min(old['end'],s['end'])-max(old['start'],s['start']))
            if overlap<3:continue
            for gid in owners[old['id']]:votes[gid]+=overlap
        if not votes:unassigned+=1;continue
        ranked=sorted(votes.items(),key=lambda item:(-item[1],item[0]))
        if len(ranked)>1 and ranked[1][1]>=ranked[0][1]*.8:
            unassigned+=1;continue
        if s['direction']=='horizontal':xy=[[s['start'],s['axis']],[s['end'],s['axis']]]
        else:xy=[[s['axis'],s['start']],[s['axis'],s['end']]]
        groups[ranked[0][0]]['export_edges_image_xy'].append(xy)
    result['export_geometry_summary']=dict(mode='green_wires_split_at_observed_junctions',
        raw_green_segments=len(observed),split_segments=len(split),unassigned_segments=unassigned,
        blue_text_gaps_joined=text_joins,
        red_junction_endpoint_snaps=junction_snaps,
        assigned_segments=sum(len(g['export_edges_image_xy']) for g in groups.values()),
        electrical_groups_unchanged=True)
    return result
