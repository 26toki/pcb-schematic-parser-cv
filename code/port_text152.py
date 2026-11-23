"""Prefer a complete aligned OCR word over its clipped local crop fragment."""
def complete_local_labels(labels,axis,side):
    local=[p for p in labels if p[1].get('source')=='port_local_ocr']
    if not local:return labels
    result=[];along=(0,2) if axis==0 else (1,3);across=(1,3) if axis==0 else (0,2)
    for score,t in local:
        b=t['bbox_xyxy'];text=t['text'].strip();options=[]
        for candidate_score,q in labels:
            full=q['text'].strip();f=q['bbox_xyxy']
            if q.get('source')=='port_local_ocr' or q.get('confidence',0)<.95 or len(full)<=len(text):continue
            if not (full.startswith(text) if side==0 else full.endswith(text)):continue
            height=min(b[across[1]]-b[across[0]],f[across[1]]-f[across[0]])
            if height<=0:continue
            if abs((b[across[0]]+b[across[1]]-f[across[0]]-f[across[1]])/2)>.25*height:continue
            anchor=along[0] if side==0 else along[1]
            if abs(b[anchor]-f[anchor])>max(3,.25*height):continue
            overlap=max(0,min(b[along[1]],f[along[1]])-max(b[along[0]],f[along[0]]))
            if overlap<.8*(b[along[1]]-b[along[0]]):continue
            options.append((candidate_score,q))
        names={q['text'] for _,q in options}
        if len(names)==1:result.append(min(options,key=lambda pair:pair[0]))
        elif not options:result.append((score,t))
        # Multiple full completions are ambiguous; do not select a fragment.
    return result
