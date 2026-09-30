"""Read-only device inventory. Does not install software, set clocks or move robots."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(argv):
    try:
        result=subprocess.run(argv,text=True,capture_output=True,timeout=20,check=False)
        return {'argv':argv,'returncode':result.returncode,'stdout':result.stdout.strip(),
                'stderr':result.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'argv':argv,'error':str(exc)}


def collect():
    result={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'system':platform.system(),
            'architecture':platform.machine(),'python':sys.version,
            'is_jetson_candidate':Path('/etc/nv_tegra_release').is_file(),
            'note':'Read-only software inventory; no inference, no benchmarks, no robot control',
            'packages':{}}
    for name in ['torch','torchvision','ultralytics','onnx','onnxruntime','tensorrt']:
        try:result['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result['packages'][name]=None
    for label,path in [('jetson_release','/etc/nv_tegra_release'),('device_model','/proc/device-tree/model')]:
        p=Path(path)
        if p.is_file():result[label]=p.read_text().replace('\x00','').strip()
    executable=shutil.which('trtexec')
    fallback=Path('/usr/src/tensorrt/bin/trtexec')
    if not executable and fallback.is_file():executable=str(fallback)
    result['trtexec']=executable
    result['nvpmodel']=shutil.which('nvpmodel')
    result['tegrastats']=shutil.which('tegrastats')
    if result['is_jetson_candidate']:
        result['dpkg']=command(['dpkg-query','-W','-f=${Package} ${Version}\n','nvidia-jetpack','nvidia-l4t-core','libnvinfer*'])
        if result['nvpmodel']:result['power_mode_query']=command([result['nvpmodel'],'-q'])
        if executable:result['trtexec_help']=command([executable,'--help'])
        try:
            import torch
            result['cuda_available']=torch.cuda.is_available()
            if result['cuda_available']:result['cuda_device_name']=torch.cuda.get_device_name(0)
        except Exception as exc:result['cuda_check_error']=str(exc)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=collect()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2)
    print('Inventory saved:',args.output,'Jetson candidate:',result['is_jetson_candidate'])
