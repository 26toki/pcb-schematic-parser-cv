"""Merge supported ordinary labels, never invent inter-island wire segments."""
import copy,re
def power_label(text):
    return bool(re.fullmatch(r'(?:[+\-]?\d+(?:\.\d+)?V\d*|[ADP]?GND\w*|GROUND|[ADP]?V(?:CC|DD|SS|IN|OUT|BAT|REF|BUS)\w*|BAT(?:TERY)?\w*)',text.strip(),re.I))

def merge_ordinary_labels(candidate):
    result=copy.deepcopy(candidate);groups=result['networks']['networks'];parents=list(range(len(groups)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    labels={c['id']:c for c in result['passives'] if c.get('accepted') and c['type'] in ('net_input','net_output','net_short')}
    rails={str(s.get('name','')).casefold() for s in result['networks'].get('power_symbols',[])}
    blockers=[ic['body_bbox_xyxy'] for ic in result.get('ics',[])]
    blockers += [d['body_bbox_xyxy'] for d in result.get('component_associations',[]) if d.get('type')=='box' and d.get('detector_confidence',0)>=.8]
    for key,c in list(labels.items()):
        e=c.get('label_evidence',{});b=c.get('body_bbox_xyxy')
        if not e.get('row_verified') or e.get('confidence',0)<.85 or c.get('geometry_source')!='observed_pentagonal_port_single_wire_direction_and_ocr':
            labels.pop(key);continue
        if b and any(x<=(b[0]+b[2])/2<=u and y<=(b[1]+b[3])/2<=v for x,y,u,v in blockers):labels.pop(key)
    seen={};events=[]
    for i,g in enumerate(groups):
        for pin in g['pins']:
            if pin.get('kind')!='passive':continue
            c=labels.get(pin.get('component_id'))
            if c is None:continue
            # Value contains observed text; generated designator suffix is not a label.
            text=c.get('value')
            if not isinstance(text,str) or not text or text!=text.strip() or power_label(text) or text.casefold() in rails:continue
            if text in seen:
                j=seen[text];a,b=find(i),find(j)
                if a!=b:parents[a]=b;events.append(dict(label=text,groups=[groups[j]['id'],g['id']]))
            else:seen[text]=i
    buckets={}
    for i,g in enumerate(groups):buckets.setdefault(find(i),[]).append(g)
    combined=[]
    for bucket in buckets.values():
        first=copy.deepcopy(bucket[0])
        if len(bucket)>1:
            pins=[];keys=set()
            for g in bucket:
                for p in g['pins']:
                    key=(p.get('kind','ic'),p.get('ic_id') or p.get('component_id'),p['number'])
                    if key not in keys:keys.add(key);pins.append(copy.deepcopy(p))
            first['pins']=pins;first['segment_ids']=list(dict.fromkeys(s for g in bucket for s in g['segment_ids']))
            # Preserve original per-island geometry; missing cached edges are
            # left for the existing exporter to rebuild from segment IDs.
            if all(g.get('export_edges_image_xy') is not None for g in bucket):first['export_edges_image_xy']=[e for g in bucket for e in g['export_edges_image_xy']]
            else:first.pop('export_edges_image_xy',None)
            first['logical_merge_evidence']=dict(labels=[e['label'] for e in events if any(x in [g['id'] for g in bucket] for x in e['groups'])],physical_group_ids=[g['id'] for g in bucket])
        combined.append(first)
    result['networks']['networks']=combined
    return result,events
