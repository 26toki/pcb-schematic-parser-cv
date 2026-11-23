"""Batch image inference to INTERNAL candidates. Not a final competition exporter."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCRIPTS=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stage(python,script,args,folder,name):
    begin=time.perf_counter()
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUTF8='1')
    # This pipeline performs no API calls. Do not give its workers any API credentials.
    for key in ('LLM_API_KEY','DASHSCOPE_API_KEY','OPENAI_API_KEY'):
        env.pop(key,None)
    with (folder/(name+'.log')).open('w',encoding='utf-8') as log:
        result=subprocess.run([str(python),str(SCRIPTS/script),*[str(v) for v in args]],
                              stdout=log,stderr=subprocess.STDOUT,env=env,timeout=1200)
    if result.returncode:
        raise RuntimeError(f'{name} failed (exit {result.returncode}); see {folder/(name+".log")}')
    return dict(stage=name,seconds=round(time.perf_counter()-begin,3))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('-input',required=True,type=Path)
    p.add_argument('-output',required=True,type=Path)
    for key in ('model','library','priors'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--pin-priors',type=Path,help='Optional frozen training-derived r/c tip offsets')
    p.add_argument('--symbol-pin-priors',type=Path,help='Optional voltage offsets and IC tip intervals')
    p.add_argument('--recover-visible-pins',action='store_true',help='Recover numeric library-out pins from localized glyphs')
    p.add_argument('--recover-vertical-pins',action='store_true',help='Recover library-supported top/bottom pins with rotated OCR')
    p.add_argument('--detector-python',default=sys.executable,type=Path)
    p.add_argument('--analysis-python',default=sys.executable,type=Path)
    p.add_argument('--device',default=None)
    a=p.parse_args()
    if a.pin_priors is not None:
        a.pin_priors=a.pin_priors.resolve()
        if not a.pin_priors.is_file():p.error('Missing pin priors')
    if a.symbol_pin_priors is not None:
        a.symbol_pin_priors=a.symbol_pin_priors.resolve()
        if not a.symbol_pin_priors.is_file():p.error('Missing symbol pin priors')
    for key in ('input','output','model','library','priors','detector_python','analysis_python'):
        setattr(a,key,getattr(a,key).resolve())
    if not a.input.is_dir():p.error('Input must be a directory')
    if a.output==a.input or a.input in a.output.parents or a.output in a.input.parents:
        p.error('Input and output must be separate non-nested directories')
    if a.output.exists() and any(a.output.iterdir()):p.error('Use an empty output directory to avoid mixing runs')
    for key in ('model','library','priors','detector_python','analysis_python'):
        if not getattr(a,key).is_file():p.error(f'Missing {key}')
    images=sorted((f for f in a.input.iterdir() if f.is_file() and f.suffix.lower()=='.png'),key=lambda f:f.name)
    if not images:p.error('No PNG images found')
    if len({f.stem.casefold() for f in images})!=len(images):p.error('Case stems must be unique')
    a.output.mkdir(parents=True,exist_ok=True)
    provenance={key:dict(path=str(getattr(a,key)),sha256=sha(getattr(a,key))) for key in ('model','library','priors')}
    if a.pin_priors is not None:
        provenance['pin_priors']=dict(path=str(a.pin_priors),sha256=sha(a.pin_priors))
    if a.symbol_pin_priors is not None:
        provenance['symbol_pin_priors']=dict(path=str(a.symbol_pin_priors),sha256=sha(a.symbol_pin_priors))
    results=[];begin=time.perf_counter()
    for image in images:
        name=image.stem;folder=a.output/'runs'/name;folder.mkdir(parents=True)
        item=dict(case_name=name,image=str(image),image_sha256=sha(image),status='running',stages=[])
        print(f'[{name}] starting',flush=True)
        def stage(script,args,label,detector=False):
            item['stages'].append(run_stage(a.detector_python if detector else a.analysis_python,script,args,folder,label))
            print(f'[{name}] {label} complete',flush=True)
        try:
            det=folder/'detector'/'detections.json';ocr=folder/'ocr'/'full'/'ocr.json'
            assoc=folder/'association'/'associations.json'
            pins=folder/'number_rows'/name/'pins.json';passives=folder/'passives'/'passives.json'
            net=folder/'networks'/'wire_networks.json';power=folder/'power'/'wire_networks.json'
            args=['--image',image,'--model',a.model,'--priors',a.priors,'--output',det.parent]
            if a.device is not None:args+=['--device',a.device]
            stage('detect_image_candidates.py',args,'detector',True)
            stage('read_schematic_text.py',['--image',image,'--output',folder/'ocr','--full-only'],'ocr')
            stage('associate_detector_text.py',['--image',image,'--detections',det,'--ocr',ocr,
                  '--library',a.library,'--output',assoc.parent,'--variant','filtered'],'association')
            stage('locate_detector_ic_pins.py',['--image',image,'--associations',assoc,'--ocr',ocr,
                  '--library',a.library,'--output',folder/'ic_source'/name,'--local-ocr'],'ic_contacts')
            stage('recognize_pin_number_rows.py',['--source',folder/'ic_source','--library',a.library,
                  '--output',folder/'number_rows','--cases',name],'pin_numbers')
            if a.recover_visible_pins:
                recovered=folder/'visible_pins'/'pins.json'
                stage('recover_visible_ic_pins.py',['--pins',pins,'--local-ocr',folder/'ic_source'/name/'local_ocr',
                      '--library',a.library,'--output',recovered.parent],'visible_pin_recovery')
                pins=recovered
            if a.recover_vertical_pins:
                recovered=folder/'vertical_pins'/'pins.json'
                stage('recover_vertical_ic_pins.py',['--pins',pins,'--library',a.library,'--output',recovered.parent],'vertical_pin_recovery')
                pins=recovered
            if a.recover_visible_pins:
                refined=folder/'refined_pins'/'pins.json'
                stage('refine_ic_pin_rows.py',['--pins',pins,'--local-ocr',folder/'ic_source'/name/'local_ocr',
                      '--library',a.library,'--full-ocr',ocr,'--output',refined.parent],'physical_pin_row_refinement')
                pins=refined
            stage('observe_passive_terminals.py',['--associations',assoc,'--ocr',ocr,
                  '--output',passives.parent,'--rescue-ocr'],'passive_contacts')
            tips=folder/'passive_tips'/'pin_tips.json'
            if a.pin_priors is not None:
                stage('locate_passive_pin_tips.py',['--passives',passives,'--priors',a.pin_priors,
                      '--output',tips.parent],'passive_pin_tips')
            stage('trace_ic_wire_networks.py',['--pins',pins,'--associations',assoc,
                  '--passives',passives,'--ocr',ocr,'--output',net.parent],'wire_graph')
            stage('attach_power_symbols.py',['--networks',net,'--pins',pins,'--detections',det,
                  '--ocr',ocr,'--output',power.parent],'power_contacts')
            symbol_tips=folder/'symbol_tips'/'symbol_pin_tips.json'
            if a.symbol_pin_priors is not None:
                stage('locate_symbol_pin_tips.py',['--pins',pins,'--networks',power,
                      '--priors',a.symbol_pin_priors,'--output',symbol_tips.parent],'symbol_pin_tips')
            ic=read(pins);pa=read(passives);nw=read(power);ass=read(assoc)
            summaries=dict(ic=ic['summary'],passives=pa['summary'],networks=nw['summary'],association=ass['summary'])
            if a.symbol_pin_priors is not None:
                symbol_report=read(symbol_tips);ic['ics']=symbol_report['ics']
                summaries['symbol_pin_tips']=symbol_report['summary']
                nw['power_symbols']=symbol_report['power_symbols']
                by_power={s['component_id']:s for s in nw['power_symbols']}
                for group in nw['networks']:
                    for contact in group['pins']:
                        if contact.get('kind')!='power':continue
                        symbol=by_power[contact['component_id']]
                        contact['pin_tip_status']=symbol['pin_tip_status']
                        if symbol['pin_tip_candidate'] is not None:
                            contact['proposed_pin_position_image_xy']=symbol['pin_tip_candidate']['position_image_xy']
            if a.pin_priors is not None:
                tip_report=read(tips);pa['components']=tip_report['components']
                summaries['passive_pin_tips']=tip_report['summary']
                by_terminal={(c['id'],t['internal_terminal']):t for c in pa['components']
                             if c['pin_tip_status']=='two_image_supported_offset_hypotheses'
                             for t in c['pin_tip_candidates']}
                for network in nw['networks']:
                    for contact in network['pins']:
                        if contact.get('kind')!='passive':continue
                        tip=by_terminal.get((contact['component_id'],contact['number']))
                        if tip:
                            contact.update(proposed_pin_number=tip['number'],
                                           proposed_pin_position_image_xy=tip['position_image_xy'],
                                           pin_mapping_status='training_convention_hypothesis')
            from wire_export_geometry import attach_export_geometry
            nw=attach_export_geometry(nw,image)
            candidate=dict(schema_version='pcb-internal-candidates-v1',case_name=name,
                           coordinate_system='image pixels; origin top-left; x right; y down',
                           submission_ready=False,provenance=provenance,input=item,
                           summary=summaries,ics=ic['ics'],passives=pa['components'],
                           networks=nw,component_associations=ass['predictions'],
                           unsupported_detections=ass['unsupported_detections'],
                           limitations=['Body contacts are not final symbol pin positions.',
                                        'Passive terminal order is internal, not official numbering.',
                                        'Other component classes and some contacts remain unresolved.',
                                        'No LLM review applied; no official scoring or final JSON export.'])
            from logical_labels146 import merge_ordinary_labels
            candidate,logical_events=merge_ordinary_labels(candidate)
            write(folder/'logical_label_merges.json',dict(events=logical_events,physical_group_count=len(nw['networks']),logical_group_count=len(candidate['networks']['networks'])))
            item.update(status='complete',summary=summaries)
            candidate['input']=item
            write(a.output/'candidates'/(name+'.json'),candidate)
        except Exception as exc:
            item.update(status='failed',error=str(exc))
            print(f'[{name}] failed: {exc}',flush=True)
        write(folder/'case_report.json',item);results.append(item)
    complete=sum(r['status']=='complete' for r in results)
    write(a.output/'decision_log.json',dict(method='yolo_ocr_geometry_candidate_pipeline',
          timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),processed_cases=complete,
          token_usage=dict(total_tokens=0,prompt_tokens=0,completion_tokens=0,api_calls=0)))
    report=dict(submission_ready=False,total_cases=len(images),completed_cases=complete,
                failed_cases=len(images)-complete,seconds=round(time.perf_counter()-begin,3),
                provenance=provenance,cases=results,
                pipeline_options=dict(visible_pin_recovery=a.recover_visible_pins,vertical_pin_recovery=a.recover_vertical_pins),
                note='Internal candidates only. No final_results/*_Pred.json until final coordinates/schema are resolved.')
    write(a.output/'batch_report.json',report)
    print(json.dumps({k:report[k] for k in ('total_cases','completed_cases','failed_cases','seconds')}),flush=True)
    return 0 if complete==len(images) else 1


if __name__=='__main__':sys.exit(main())
