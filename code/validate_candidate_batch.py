"""Validate batch artifacts and provenance without loading any GT annotations."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(root):
    report=read(root/'batch_report.json');decision=read(root/'decision_log.json')
    assert report['completed_cases']+report['failed_cases']==report['total_cases']
    assert decision['processed_cases']==report['completed_cases']
    assert all(value==0 for value in decision['token_usage'].values())
    for asset in report['provenance'].values():assert sha(asset['path'])==asset['sha256']
    counts=[]
    for case in report['cases']:
        assert sha(case['image'])==case['image_sha256']
        if case['status']!='complete':continue
        candidate=read(root/'candidates'/(case['case_name']+'.json'))
        assert candidate['submission_ready'] is False
        assert candidate['input']['status']=='complete'
        options=report.get('pipeline_options',{})
        refinement=int(bool(options.get('visible_pin_recovery',False)))
        assert len(case['stages'])==8+int('pin_priors' in report['provenance'])+int('symbol_pin_priors' in report['provenance'])+sum(int(v) for v in options.values())+refinement
        assert sum(s['stage']=='physical_pin_row_refinement' for s in case['stages'])==refinement
        width,height=Image.open(case['image']).size
        def point(xy):assert len(xy)==2 and 0<=xy[0]<=width and 0<=xy[1]<=height,xy
        def box(b):
            point(b[:2]);point(b[2:]);assert b[0]<=b[2] and b[1]<=b[3]
        icids=[ic['detection_id'] for ic in candidate['ics']]
        paids=[pa['id'] for pa in candidate['passives']]
        proposed_tips={(c['id'],pin['internal_terminal']):pin for c in candidate['passives']
                       if c.get('pin_tip_status')=='two_image_supported_offset_hypotheses'
                       for pin in c['pin_tip_candidates']}
        assert len(icids)==len(set(icids)) and len(paids)==len(set(paids))
        for ic in candidate['ics']:
            box(ic['body_bbox_xyxy'])
            for pin in ic['pins']:
                for field in ('body_contact_xy','pin_position'):
                    if pin[field] is not None:point(pin[field])
                if pin.get('tip_geometry',{}).get('status')=='ambiguous_pin_wire_boundary':
                    assert pin['tip_geometry']['position_image_xy'] is None
                    for proposed in pin['tip_geometry']['supported_positions']:point(proposed['position_image_xy'])
        for pa in candidate['passives']:
            box(pa['body_bbox_xyxy'])
            assert len(pa['terminal_contacts_xy'])==(1 if pa['type'] in ('net_input','net_output') else 2)
            for xy in pa['terminal_contacts_xy']:point(xy)
            if pa.get('pin_tip_status')=='two_image_supported_offset_hypotheses':
                assert len(pa['pin_tip_candidates'])==2
                assert {p['number'] for p in pa['pin_tip_candidates']}=={'1','2'}
                assert pa['terminal_number_mapping']=={p['internal_terminal']:p['number'] for p in pa['pin_tip_candidates']}
                for pin in pa['pin_tip_candidates']:point(pin['position_image_xy'])
        nw=candidate['networks'];segmentids=[s['id'] for s in nw['segments']]
        power_tips={s['component_id']:s for s in nw['power_symbols']}
        for symbol in power_tips.values():
            if 'pin_tip_candidate' not in symbol:continue
            if symbol['type']=='gnd':assert symbol['pin_tip_candidate'] is None
            if symbol['pin_tip_candidate'] is not None:point(symbol['pin_tip_candidate']['position_image_xy'])
        assert len(segmentids)==len(set(segmentids))
        netids=[n['id'] for n in nw['networks']];assert len(netids)==len(set(netids))
        for n in nw['networks']:
            assert set(n['segment_ids'])<=set(segmentids)
            for pin in n['pins']:
                point(pin['contact_xy']);assert pin['segment_id'] in n['segment_ids']
                kind=pin.get('kind','ic')
                if kind=='ic':assert pin['ic_id'] in icids
                elif kind=='passive':assert pin['component_id'] in paids
                else:assert kind=='power'
                if 'proposed_pin_position_image_xy' in pin:
                    point(pin['proposed_pin_position_image_xy'])
                    if kind=='passive':
                        assert pin['proposed_pin_number'] in ('1','2')
                        tip=proposed_tips[(pin['component_id'],pin['number'])]
                        assert pin['proposed_pin_number']==tip['number']
                        assert pin['proposed_pin_position_image_xy']==tip['position_image_xy']
                    else:
                        assert kind=='power'
                        assert pin['proposed_pin_position_image_xy']==power_tips[pin['component_id']]['pin_tip_candidate']['position_image_xy']
        counts.append(dict(case_name=case['case_name'],ics=len(icids),
                           accepted_passives=sum(p['accepted'] for p in candidate['passives']),
                           ic_text_pairs=sum(len(ic['pins']) for ic in candidate['ics']),
                           power_symbols=len(nw['power_symbols']),wire_groups=len(netids)))
    result=dict(valid=True,completed_cases=report['completed_cases'],failed_cases=report['failed_cases'],
                seconds=report['seconds'],cases=counts,
                note='Internal schema/provenance validation only; not official scoring or submission validation.')
    (root/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(validate(p.parse_args().output),ensure_ascii=False))
