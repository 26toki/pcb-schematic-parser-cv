"""PNG-only YOLO worker. No annotations, dataset manifests or API inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('image','model','priors','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default=None)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str((a.output/'settings').resolve())
    os.environ['MPLCONFIGDIR']=str((a.output/'matplotlib').resolve())
    os.environ['YOLO_AUTOINSTALL']='false'
    import torch
    from ultralytics import settings
    from trusted_yolo import load_verified_yolo
    from inference_core import predict,refine,merge_refined
    from inference_core import filter_candidates
    settings.update({'sync':False});torch.set_num_threads(2)
    rgba=Image.open(a.image).convert('RGBA')
    image=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
    model=load_verified_yolo(a.model);priors=json.loads(a.priors.read_text(encoding='utf-8'))
    full=predict(model,image,device=a.device)
    tiled=predict(model,image,True,device=a.device)
    merged=merge_refined(refine(tiled,priors,*image.size))
    kept,suppressed,flagged=filter_candidates(merged,full)
    report=dict(image=str(a.image.resolve()),versions={'full':full,'filtered':kept},
                suppressed=suppressed,flagged_ids=flagged,
                provenance={key+'_sha256':hashlib.sha256(path.read_bytes()).hexdigest()
                            for key,path in [('image',a.image),('model',a.model),('priors',a.priors)]})
    (a.output/'detections.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(full=len(full),filtered=len(kept),suppressed=len(suppressed))))


if __name__=='__main__':main()
