"""Record actual base versions and test required functions, not exact vendor versions."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys


def main():
    import torch,torchvision,cv2,numpy,onnxruntime,ultralytics
    from rapidocr_onnxruntime import RapidOCR
    from trusted_yolo import load_verified_yolo
    packages=('torch','torchvision','opencv-python-headless','opencv-python','numpy','ultralytics','rapidocr-onnxruntime','onnxruntime')
    versions={}
    for package in packages:
        try:versions[package]=metadata.version(package)
        except metadata.PackageNotFoundError:versions[package]=None
    actual=dict(python=sys.version.split()[0],cuda_build=torch.version.cuda,cv2=cv2.__version__,versions=versions)
    print(json.dumps(dict(environment=actual)),flush=True)
    if sys.version_info[:2]!=(3,12):raise RuntimeError('Packaged wheels require Python 3.12')
    boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.],[20.,20.,30.,30.]])
    keep=torchvision.ops.nms(boxes,torch.tensor([.9,.8,.7]),.5)
    if keep.tolist()!=[0,2]:raise RuntimeError('torchvision NMS functional check failed')
    resized=cv2.resize(numpy.zeros((8,8,3),dtype=numpy.uint8),(16,16))
    if resized.shape!=(16,16,3):raise RuntimeError('OpenCV functional check failed')
    engine=RapidOCR()
    root=Path(__file__).resolve().parent.parent
    model=load_verified_yolo(root/'assets/yolo.pt')
    prediction=model.predict(numpy.zeros((128,128,3),dtype=numpy.uint8),device='cpu',imgsz=128,verbose=False)
    if len(prediction)!=1:raise RuntimeError('YOLO CPU prediction failed')
    print(json.dumps(dict(functional_checks_passed=True,checks=['torchvision_cpu_nms','opencv_resize','ocr_init','verified_yolo_load','yolo_cpu_inference'],
                         gpu_available=torch.cuda.is_available(),environment=actual)),flush=True)


if __name__=='__main__':main()
