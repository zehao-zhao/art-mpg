"""Run all 180 R1/R2 jobs sequentially on a single 16-core lease."""
import argparse,json
from pathlib import Path
from run_render import run
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--references',type=Path,required=True,help='JSON mapping scene names to a list of reference EXRs');p.add_argument('--models',type=Path);p.add_argument('--cpus',required=True);a=p.parse_args()
 models=json.loads(a.models.read_text()) if a.models else {}
 refs=json.loads(a.references.read_text());jobs=json.loads((Path(__file__).resolve().parent/'jobs.json').read_text());cpus=list(map(int,a.cpus.split(',')))
 for j in jobs:
  out=a.output/f"{j['scene']}__{j['method']}__r{j['seed']}__t{j['budget_seconds']}"
  if (out/'result.json').is_file():continue
  run(j['scene'],j['method'],j['seed'],j['budget_seconds'],cpus,a.build,out,refs[j['scene']],models.get(j['scene'],{}).get(j['method']))
