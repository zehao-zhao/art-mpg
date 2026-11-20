"""Run the recorded 4200-job quantitative matrix, or all D01 native crops."""
import argparse,json
from pathlib import Path
from run_render import run,HERE
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu',type=int,required=True);p.add_argument('--references',type=Path,help='JSON: scene -> 64px reference EXR');p.add_argument('--native-crops',action='store_true');a=p.parse_args()
 refs=json.loads(a.references.read_text()) if a.references else {};jobs=json.loads((HERE/'jobs.json').read_text())
 if a.native_crops:jobs=[dict(j,crop=c) for j in jobs if j['D_label']=='D01' and j['SPP']==60 for c in ['crop1','crop2']]
 elif not refs:p.error('--references is required for quantitative runs')
 for j in jobs:
  crop=j.get('crop');out=a.output/f"{j['scene']}__{j['method']}__{j['D_label']}__spp{j['SPP']}"+(f'__{crop}' if crop else '')
  if (out/'result.json').is_file():continue
  run(j['scene'],j['method'],j['seed'],j['model_seed'],j['SPP'],a.binary,out,a.cpu,refs.get(j['scene']),crop)
