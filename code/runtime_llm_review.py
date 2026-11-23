"""Optional bounded runtime review; proposals never mutate predictions."""
import json
from pathlib import Path
import subprocess


def merge_review_usage(decision_path, api_output):
    decision_path=Path(decision_path); api_output=Path(api_output)
    usage=json.loads((api_output/'review_usage.json').read_text(encoding='utf-8'))
    report=json.loads((api_output/'report.json').read_text(encoding='utf-8'))
    if report['mode']!='execute':raise ValueError('Actual runtime report required')
    for key in ('prompt_tokens','completion_tokens','total_tokens','api_calls','usage_unknown_calls'):
        if isinstance(usage[key],bool) or not isinstance(usage[key],int) or usage[key]<0:raise ValueError('Invalid usage')
    if usage['total_tokens']!=usage['prompt_tokens']+usage['completion_tokens']:raise ValueError('Usage totals')
    if usage['usage_unknown_calls']>usage['api_calls']:raise ValueError('Unknown usage count')
    decision=json.loads(decision_path.read_text(encoding='utf-8'))
    if decision.get('runtime_review_accounted'):raise ValueError('Runtime review already counted')
    for key in ('prompt_tokens','completion_tokens','total_tokens','api_calls'):
        decision['token_usage'][key]+=usage[key]
    decision['runtime_review_accounted']=True
    decision['runtime_review']={
        'usage_complete':usage['usage_unknown_calls']==0,
        'usage_unknown_calls':usage['usage_unknown_calls'],
        'validated_reviews':len(report['reviews']),
        'proposals_applied':0,
        'stopped':report['stopped'],
        'note':'Token sums include known usage only; unknown usage is explicitly recorded. Proposals are not applied.'}
    decision['method']='yolo_ocr_geometry_with_llm_review_proposals'
    decision_path.write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
    return dict(status='stopped' if report['stopped'] else 'complete',api_calls=usage['api_calls'],
                usage_complete=usage['usage_unknown_calls']==0,validated_reviews=len(report['reviews']),proposals_applied=0)


def run_runtime_review(code, python, work, output, mode, max_calls, max_tokens, spend_stop, reserve, log):
    if mode=='off':return dict(status='off',api_calls=0,usage_complete=True,proposals_applied=0)
    code=Path(code);work=Path(work);output=Path(output)
    queues=output/'runs'/'llm_queues';api=output/'runs'/'llm_client'
    queue_result=subprocess.run([str(python),str(code/'prepare_candidate_review_queue.py'),
        '--candidates',str(work/'candidates'),'--output',str(queues),'--max-selected-per-case','3'],stdout=log,stderr=subprocess.STDOUT)
    if queue_result.returncode:raise RuntimeError('Review queue preparation failed')
    paths=sorted(queues.glob('*/review_queue.json'))
    selected=sum(len(json.loads(path.read_text(encoding='utf-8'))['selected_task_ids']) for path in paths)
    if not selected:return dict(status='no_tasks',api_calls=0,usage_complete=True,proposals_applied=0)
    args=[str(python),str(code/'candidate_review_client.py'),'--queues',*[str(path) for path in paths],
        '--output',str(api),'--max-calls',str(max_calls),'--max-output-tokens',str(max_tokens),
        '--spend-stop-cny',str(spend_stop),'--reserve-per-call-cny',str(reserve)]
    if mode=='execute':args.append('--execute')
    result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
    if mode=='plan':
        if result.returncode:raise RuntimeError('Review plan failed')
        return dict(status='planned_not_sent',api_calls=0,usage_complete=True,proposals_applied=0)
    if not (api/'review_usage.json').exists():raise RuntimeError('Review client failed without usage report')
    summary=merge_review_usage(output/'decision_log.json',api)
    summary['client_exit_code']=result.returncode
    return summary
