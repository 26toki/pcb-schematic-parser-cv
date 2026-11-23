"""Load only the packaged, hash-verified full-module YOLO checkpoint."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

LOCAL_BASELINE_SHA256='3b05ddfdbb9baf864f8f6d30772d7bac005cd2180c793e2c6b34ccef204bfb52'


def load_verified_yolo(path,expected_sha256=None):
    import torch
    from ultralytics import YOLO
    target=Path(path).resolve()
    if expected_sha256 is None:
        root=Path(__file__).resolve().parent.parent;manifest=root/'bundle_manifest.json'
        if manifest.is_file():
            if target!=(root/'assets/yolo.pt').resolve():raise ValueError('Checkpoint is outside packaged assets')
            expected_sha256=json.loads(manifest.read_text(encoding='utf-8'))['files']['assets/yolo.pt']
        else:expected_sha256=LOCAL_BASELINE_SHA256
    if hashlib.sha256(target.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('Checkpoint hash mismatch')
    original=torch.load
    def verified_load(file,*args,**kwargs):
        if not isinstance(file,(str,os.PathLike)) or Path(file).resolve()!=target:
            raise ValueError('Unexpected checkpoint requested')
        # Ultralytics 8.3.40 stores nn.Module objects; torch>=2.6 changed the default.
        # Scope this to one immutable verified asset, never a global environment override.
        kwargs.setdefault('weights_only',False)
        return original(file,*args,**kwargs)
    with patch.object(torch,'load',verified_load):
        return YOLO(str(target))
