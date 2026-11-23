"""Export reviewable nested-schema drafts, never choose ambiguous IC tips or silently truncate nets."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
from PIL import Image
from observed_resistor_bbox168 import use_observed_body

TYPES={'box':'ic','r':'resistor','c':'capacitor','l':'inductor','v':'vcc','gnd':'gnd','d':'diode','led':'led','net_input':'net_input','net_output':'net_output'}


def point(xy,height):return dict(x=float(xy[0]),y=float(height-xy[1]))
def bbox(b,height):return [float(b[0]),float(height-b[3]),float(b[2]),float(height-b[1])]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(result,width,height,profile):
    if set(result)!= {'components','pins','nets'}:raise ValueError('Unexpected top-level fields')
    for key in result:
        if not isinstance(result[key],dict):raise ValueError('Expected objects')
    def xy(p):
        if set(p)!= {'x','y'}:raise ValueError('Point schema')
        for k,limit in [('x',width),('y',height)]:
            v=p[k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=limit:
                raise ValueError('Point range or number')
    name_key='Name' if profile=='gt-short' else 'name'
    allowed=set(TYPES) if profile=='gt-short' else set(TYPES.values())
    refs=set()
    for cid,c in result['components'].items():
        if not isinstance(cid,str) or not cid or any(ch in cid for ch in '.,()'):raise ValueError('Component identifier')
        if set(c)!= {'type',name_key,'value','bbox'} or c['type'] not in allowed:raise ValueError('Component schema')
        if c[name_key] is not None and not isinstance(c[name_key],str):raise ValueError('Name')
        if c['value'] is not None and not isinstance(c['value'],str):raise ValueError('Value')
        b=c['bbox']
        if not isinstance(b,list) or len(b)!=4:raise ValueError('Bbox schema')
        xy(dict(x=b[0],y=b[1]));xy(dict(x=b[2],y=b[3]))
        if b[0]>=b[2] or b[1]>=b[3]:raise ValueError('Bbox order')
    for cid,pins in result['pins'].items():
        if cid not in result['components'] or not isinstance(pins,dict):raise ValueError('Orphan pins')
        for pid,pin in pins.items():
            if not pid.startswith('pin_') or not pid[4:] or any(ch in pid[4:] for ch in '.,()'):raise ValueError('Pin identifier')
            if set(pin)!= {'pinname','point'} or not isinstance(pin['pinname'],str):raise ValueError('Pin schema')
            xy(pin['point']);refs.add(cid+'.'+pid[4:])
    claimed=set()
    for net in result['nets'].values():
        if set(net)!= {'hyperGraph','edges'}:raise ValueError('Net schema')
        graph=net['hyperGraph']
        if not isinstance(graph,str) or not graph.startswith('(') or not graph.endswith(')'):raise ValueError('Hypergraph format')
        members=[r.strip() for r in graph[1:-1].split(',') if r.strip()]
        if len(members)<2 or len(members)!=len(set(members)) or not set(members)<=refs:raise ValueError('Hypergraph references')
        if claimed.intersection(members):raise ValueError('Pin belongs to multiple physical nets')
        claimed.update(members)
        if not isinstance(net['edges'],dict):raise ValueError('Edges object')
        for edge in net['edges'].values():
            if not isinstance(edge,list) or len(edge)!=2:raise ValueError('Two-point edges only')
            for p in edge:xy(p)
    return dict(components=len(result['components']),pins=len(refs),nets=len(result['nets']),
                edges=sum(len(n['edges']) for n in result['nets'].values()))


def export(candidate,size,profile,priors,scoring_policy='legacy-strict'):
    width,height=size;result=dict(components={},pins={},nets={});issues=[];mapping={};sources={}
    name_key='Name' if profile=='gt-short' else 'name'
    occurrences=Counter(c['designator'] for c in candidate['ics'] if c['designator'])
    occurrences.update(c['designator'] for c in candidate['passives'] if c['accepted'] and c['designator'])
    def issue(kind,target,reason,**extra):issues.append(dict(kind=kind,target_id=target,reason=reason,**extra))
    def add_component(cid,kind,name,value,b,source):
        # Reference designators are keys, not substitute model names.
        # Network label text belongs in Name, not the electrical value field.
        if kind in ('led','switch') and name == cid:
            name = None
        if kind in ('net_input','net_output'):
            value = None
        if not cid or any(ch in cid for ch in '.,()') or cid in result['components'] or occurrences[cid]>1:
            issue('component',source,'missing_invalid_or_duplicate_designator',designator=cid);return False
        result['components'][cid]=dict(type=kind if profile=='gt-short' else TYPES[kind],
                                      **{name_key:name},value=value,bbox=bbox(b,height))
        sources[cid]=source;return True
    def add_pin(kind,source,internal,cid,number,name,xy):
        pid='pin_'+number
        if not number or any(ch in number for ch in '.,()') or pid in result['pins'].get(cid,{}):
            issue('pin',source,'invalid_or_duplicate_pin_number',number=number);return
        result['pins'].setdefault(cid,{})[pid]=dict(pinname=name,point=point(xy,height))
        mapping[(kind,source,internal)]=cid+'.'+number
    rejected_contacts=set()
    for conflict in candidate['networks'].get('contact_conflicts',[]):
        for number in conflict['candidate_numbers']:
            if number not in conflict['accepted_numbers']:
                rejected_contacts.add((conflict['ic_id'],number,tuple(conflict['contact_xy'])))
    for ic in candidate['ics']:
        cid=ic['designator'];source=ic['detection_id']
        if not add_component(cid,'box',ic['model'],None,ic['body_bbox_xyxy'],source):continue
        numbers=Counter(p['number'] for p in ic['pins'])
        for pin in ic['pins']:
            key=(source,pin['number'],tuple(pin.get('body_contact_xy') or []))
            if key in rejected_contacts:
                issue('ic_pin',source,'conflicting_contact_rejected_by_wire_graph',number=pin['number']);continue
            xy=pin.get('pin_position')
            if scoring_policy=='2026-10-14' and xy is None:
                geometry=pin.get('tip_geometry',{})
                if geometry.get('status') in ('observed_red_pin_green_wire_boundary','same_symbol_consistent_peer_length'):
                    xy=geometry['position_image_xy']
                supported=pin.get('tip_geometry',{}).get('supported_positions',[])
                if xy is None and supported:
                    # Highest training frequency among image-supported gaps; never GT-nearest.
                    xy=max(supported,key=lambda p:p['training_count'])['position_image_xy']
            if numbers[pin['number']]>1 or xy is None:
                issue('ic_pin',source,'duplicate_number_or_unresolved_tip',number=pin['number']);continue
            add_pin('ic',source,pin['number'],cid,pin['number'],pin['pinname'],xy)
    for c in candidate['passives']:
        if not c['accepted']:continue
        b=c['body_bbox_xyxy'];center=[(b[0]+b[2])/2,(b[1]+b[3])/2]
        # GT annotation boxes differ from visible plates. Use frozen TRAIN size priors.
        measured_zigzag=(c['type']=='r' and c.get('geometry_source')=='observed_alternating_zigzag_with_inline_leads')
        measured_recovery=use_observed_body(c,size)
        if c['type'] in priors and not measured_zigzag and not measured_recovery:
            sizes=[s for s in priors[c['type']] if (s['height']>=s['width'])==(c['axis']=='vertical')]
            chosen=(sizes or priors[c['type']])[0];w,h=chosen['width'],chosen['height']
            b=[max(0,center[0]-w/2),max(0,center[1]-h/2),min(width,center[0]+w/2),min(height,center[1]+h/2)]
        cid=c['designator'];source=c['id']
        if not add_component(cid,c['type'],c.get('name',cid),c['value'],b,source):continue
        if scoring_policy=='2026-10-14':
            # Positions remain required by the known JSON contract, though ordinary pins are unscored.
            # Keep distinct terminals for wire tracing; never merge through the component body.
            for i,contact in enumerate(c['terminal_contacts_xy']):
                number='1' if c['type'] in ('net_input','net_output') else str(i+1) if c['axis']=='horizontal' else str(2-i)
                add_pin('passive',source,'terminal_'+str(i+1),cid,number,number,contact)
            continue
        if c.get('pin_tip_status')!='two_image_supported_offset_hypotheses':
            issue('passive_pin',source,'unresolved_tip_geometry');continue
        for pin in c['pin_tip_candidates']:
            add_pin('passive',source,pin['internal_terminal'],cid,pin['number'],pin['number'],pin['position_image_xy'])
    counters=Counter()
    for s in candidate['networks']['power_symbols']:
        counters[s['type']]+=1;cid=s['type']+'_'+str(counters[s['type']]);source=s['component_id']
        if not add_component(cid,s['type'],s['name'],s['name'],s['body_bbox_xyxy'],source):continue
        tip=s.get('pin_tip_candidate')
        if scoring_policy=='2026-10-14' and tip is None:
            tip=dict(position_image_xy=s['contact_xy'])
        if tip is None:
            issue('power_pin',source,'unresolved_ground_or_voltage_tip');continue
        add_pin('power',source,s['number'],cid,'1','1',tip['position_image_xy'])
    segments={s['id']:s for s in candidate['networks']['segments']}
    for group in candidate['networks']['networks']:
        members=[];missing=[]
        for pin in group['pins']:
            kind=pin.get('kind','ic');source=pin['ic_id'] if kind=='ic' else pin['component_id']
            ref=mapping.get((kind,source,pin['number']))
            if ref is None:missing.append(dict(kind=kind,source=source,number=pin['number']))
            else:members.append(ref)
        if missing:
            issue('network',group['id'],'contains_unexported_pins',missing=missing);continue
        if len(set(members))<2:
            issue('network',group['id'],'fewer_than_two_exported_members');continue
        edges={}
        geometry=group.get('export_edges_image_xy')
        if geometry is None:
            geometry=[]
            for sid in group['segment_ids']:
                s=segments[sid]
                geometry.append([[s['start'],s['axis']],[s['end'],s['axis']]] if s['direction']=='horizontal' else [[s['axis'],s['start']],[s['axis'],s['end']]])
        for xy in geometry:
            edges['edge_'+str(len(edges)+1)]=[point(p,height) for p in xy]
        result['nets']['net_'+str(len(result['nets'])+1)]=dict(hyperGraph='('+','.join(sorted(set(members)))+')',edges=edges)
    return result,issues,sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=('gt-short','manual-long'),default='gt-short')
    p.add_argument('--scoring-policy',choices=('legacy-strict','2026-10-14'),default='legacy-strict')
    p.add_argument('--decision-log',type=Path,help='Optional combined log including already-counted API reviews')
    a=p.parse_args();files=sorted(a.candidates.glob('*.json'))
    if not files:p.error('No candidates found')
    if a.output.exists() and any(a.output.iterdir()):p.error('Use an empty draft output directory')
    a.output.mkdir(parents=True,exist_ok=True);cases=[]
    for path in files:
        c=json.loads(path.read_text(encoding='utf-8'));name=c['case_name']
        if name!=path.stem:raise ValueError('Case/file identity mismatch')
        image=Path(c['input']['image'])
        if sha(image)!=c['input']['image_sha256']:raise ValueError('Stale image')
        prior=c['provenance']['priors'];prior_path=Path(prior['path'])
        if sha(prior_path)!=prior['sha256']:raise ValueError('Stale size priors')
        with Image.open(image) as opened:size=opened.size
        result,issues,sources=export(c,size,a.profile,json.loads(prior_path.read_text(encoding='utf-8')),a.scoring_policy)
        counts=validate(result,*size,a.profile)
        dest=a.output/'final_results'/(name+'_Pred.json');dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        report=dict(case_name=name,profile=a.profile,scoring_policy=a.scoring_policy,submission_ready=False,counts=counts,issues=issues,
                    component_source_ids=sources,candidate_path=str(path.resolve()),candidate_sha256=sha(path),
                    output_sha256=sha(dest),coordinate_system='bottom-left origin, x right, y up',
                    limitations='Draft; not platform validated. Under 2026-10-14 policy ordinary pin positions are schema-compatible body contacts, unscored according to user update. IC points use a training-frequency estimate supported by pixels, not verified true tips. Two-level matching remains platform-only. Incomplete groups are omitted intact; edges segmentation still needs validation.')
        review=a.output/'export_reports'/(name+'.json');review.parent.mkdir(exist_ok=True)
        review.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        cases.append(dict(case_name=name,counts=counts,issues=len(issues)))
    decision=json.loads((a.decision_log or a.candidates.parent/'decision_log.json').read_text(encoding='utf-8'))
    if decision['processed_cases']!=len(files):raise ValueError('Input decision log count mismatch')
    (a.output/'decision_log.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    report=dict(submission_ready=False,profile=a.profile,scoring_policy=a.scoring_policy,cases=cases,
                note='Nested structure/reference/range checks passed locally. Not validated by official platform; not ready to submit.')
    (a.output/'draft_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
