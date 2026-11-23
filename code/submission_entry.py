"""PNG inference with optional bounded LLM review; proposals are not applied."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from runtime_llm_review import run_runtime_review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('-input',type=Path,required=True);p.add_argument('-output',type=Path,required=True)
    p.add_argument('--detector-python',type=Path,default=Path(sys.executable))
    p.add_argument('--analysis-python',type=Path,default=Path(sys.executable))
    p.add_argument('--device',default='cpu',help='Validated baseline uses CPU; pass an explicit device to opt into GPU')
    p.add_argument('--profile',choices=('gt-short','manual-long'),default='gt-short')
    p.add_argument('--llm-mode',choices=('off','plan','execute'),default=os.environ.get('PCB_LLM_MODE','off'))
    p.add_argument('--llm-max-calls',type=int,default=3)
    p.add_argument('--llm-max-output-tokens',type=int,default=768)
    p.add_argument('--llm-spend-stop-cny',type=float,default=.2)
    p.add_argument('--llm-reserve-per-call-cny',type=float,default=.05)
    a=p.parse_args();root=Path(__file__).resolve().parent.parent;code=root/'code';assets=root/'assets'
    if a.llm_mode not in ('off','plan','execute'):p.error('Invalid PCB_LLM_MODE')
    if not 1<=a.llm_max_calls<=6 or not 256<=a.llm_max_output_tokens<=2048:p.error('LLM call/token limits')
    if any(not math.isfinite(value) or value<=0 for value in (a.llm_spend_stop_cny,a.llm_reserve_per_call_cny)):p.error('LLM budget limits')
    requested_llm_mode=a.llm_mode
    missing_llm=[key for key in ('LLM_BASE_URL','LLM_API_KEY','LLM_MODEL') if not os.environ.get(key)] if a.llm_mode=='execute' else []
    if missing_llm:
        a.llm_mode='off'
        print(json.dumps(dict(llm_status='disabled_missing_configuration',missing_variables=missing_llm,
                              action='continue_cv_without_api')),flush=True)
    a.input=a.input.resolve();a.output=a.output.resolve()
    if not a.input.is_dir():p.error('Input directory does not exist')
    if a.input==a.output or a.input in a.output.parents or a.output in a.input.parents:p.error('Separate non-nested input/output directories required')
    if a.output.exists() and not a.output.is_dir():p.error('Output must be a directory')
    manifest=json.loads((root/'bundle_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        file=(root/name).resolve()
        if root not in file.parents or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError('Bundle file/hash mismatch')
    images=sorted(f for f in a.input.iterdir() if f.is_file() and f.suffix.lower()=='.png')
    if not images:p.error('No PNG input')
    if len({f.stem.casefold() for f in images})!=len(images):p.error('Duplicate case stems')
    # The evaluator may create report/log files before starting the participant.
    # Preserve those files and isolate all internal stages from prior runs.
    final=a.output/'final_results';final.mkdir(parents=True,exist_ok=True)
    runs=a.output/'runs';runs.mkdir(exist_ok=True)
    current=Path(tempfile.mkdtemp(prefix='agent-',dir=runs))
    work=current/'pipeline';exported=current/'export'
    print(json.dumps(dict(input=str(a.input),output=str(a.output),run_directory=str(current.relative_to(a.output)),
                          input_cases=len(images),device=a.device)),flush=True)
    args=[str(a.analysis_python.resolve()),str(code/'agent_pipeline.py'),'-input',str(a.input),'-output',str(work),
          '--model',str(assets/'yolo.pt'),'--library',str(assets/'BoxICLibrary.json'),'--priors',str(assets/'size_priors.json'),
          '--symbol-pin-priors',str(assets/'symbol_pin_priors.json'),'--recover-visible-pins','--recover-vertical-pins',
          '--detector-python',str(a.detector_python.resolve()),'--analysis-python',str(a.analysis_python.resolve())]
    if a.device is not None:args+=['--device',a.device]
    print('Starting packaged PNG pipeline',flush=True)
    with (final/'run.log').open('w',encoding='utf-8') as log:
        pipeline=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads((work/'batch_report.json').read_text(encoding='utf-8'))
        if report['completed_cases']:
            recovery=subprocess.run([str(a.analysis_python.resolve()),str(code/'recover_power173.py'),'--pipeline',str(work)],stdout=log,stderr=subprocess.STDOUT)
            if recovery.returncode:raise RuntimeError('Power recovery stage failed; inspect run.log')
            validate=subprocess.run([str(a.analysis_python.resolve()),str(code/'validate_candidate_batch.py'),'--output',str(work)],stdout=log,stderr=subprocess.STDOUT)
            if validate.returncode:raise RuntimeError('Internal candidate validation failed; inspect run.log')
            export=subprocess.run([str(a.analysis_python.resolve()),str(code/'export_prediction_draft.py'),'--candidates',str(work/'candidates'),
                '--output',str(exported),'--profile',a.profile,'--scoring-policy','2026-10-14'],stdout=log,stderr=subprocess.STDOUT)
            if export.returncode:raise RuntimeError('Export failed; inspect run.log')
            for file in (exported/'final_results').glob('*_Pred.json'):shutil.copyfile(file,final/file.name)
        shutil.copyfile(work/'decision_log.json',current/'decision_log.json')
        review=dict(status='no_completed_cases',api_calls=0,usage_complete=True,proposals_applied=0)
        if report['completed_cases']:
            review=run_runtime_review(code,a.analysis_python.resolve(),work,current,a.llm_mode,
                a.llm_max_calls,a.llm_max_output_tokens,a.llm_spend_stop_cny,a.llm_reserve_per_call_cny,log)
        if missing_llm:review.update(status='disabled_missing_configuration',missing_variables=missing_llm)
        shutil.copyfile(current/'decision_log.json',a.output/'decision_log.json')
        decision=json.loads((a.output/'decision_log.json').read_text(encoding='utf-8'))
        decision['llm_runtime']=dict(requested_mode=requested_llm_mode,active_mode=a.llm_mode,status=review['status'],missing_variables=missing_llm)
        (a.output/'decision_log.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
    failed=[]
    for image in images:
        if not (exported/'final_results'/(image.stem+'_Pred.json')).is_file():
            failed.append(image.stem)
            (final/(image.stem+'_Pred.json')).write_text(json.dumps(dict(components={},pins={},nets={})),encoding='utf-8')
    result=dict(format_files_complete=True,platform_validated=False,profile=a.profile,scoring_policy='2026-10-14',
                cases=len(images),completed_cases=report['completed_cases'],failed_cases=failed,api_calls=review['api_calls'],
                llm_mode=a.llm_mode,requested_llm_mode=requested_llm_mode,llm_review=review,run_directory=str(current.relative_to(a.output)),device=a.device,
                note='CV predictions with optional review records. Proposals are not applied. Historical development calls are not replayed. Official platform validation remains pending.')
    (a.output/'entry_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    # Optional API failure is explicit in logs/accounting but does not invalidate completed CV predictions.
    return 1 if failed or pipeline.returncode else 0


if __name__=='__main__':raise SystemExit(main())
