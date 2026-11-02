"""Build the independent Spark split producer against the tested Sol layout."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--kitchen',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--arch',default='86')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
source=root/'native/spark_split/producer.cu'
headers=a.kitchen/'comfy_kitchen/backends/cuda/sage_attention'
a.output.parent.mkdir(parents=True,exist_ok=True)
cmd=['nvcc','-O3','-std=c++20','--use_fast_math','-shared','-Xcompiler','-fPIC','-gencode',
     f'arch=compute_{a.arch},code=sm_{a.arch}','-I',str(headers),str(source),'-o',str(a.output)]
subprocess.run(cmd,check=True)
files=[source,headers/'sol_layout.cuh',headers/'mma.cuh',a.output]
a.output.with_suffix('.build.json').write_text(json.dumps(dict(command=cmd,sha256={
    str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}),indent=2)+'\n')
