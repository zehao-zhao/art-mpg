"""One equal-time render: joint R/T, online MPG or NPM at 256 pixels."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from render_utils import ROOT,prepare,render,mse
SCENES=['living','slabs','cornell_glass','pool','water','stone'];METHODS=['art_joint','mpg','npm']
def run(scene,method,seed,budget,cpus,build,out,references,model_override=None):
 if len(cpus)!=16:raise ValueError('R1/R2 requires 16 physical cores')
 template=Path(__file__).resolve().parent/'scenes'/f'{scene}__{method}.xml'
 mapping=json.loads((Path(__file__).resolve().parent/'scene_paths.json').read_text())
 model=model_override or (f'model_exports/equal_time/joint_rt/{scene}.txt' if method=='art_joint' else None)
 xml=prepare(template.read_text(),ROOT/mapping[scene],256,999999,seed,budget,model)
 data,log=render(Path(build)/method/'dist/mitsuba',xml,out,cpus,budget)
 if method in ('art_joint','npm') and 'masked_diffusion_runtime_v1' not in log and 'Loaded masked diffusion runtime' not in log and 'Loaded plain MLP' not in log and 'Loaded NPM' not in log:
  # Runtime implementations use several model-specific success messages.
  import re
  if not re.search(r'Loaded .*runtime|[Ll]oaded.*joint|backbone_type=joint_categorical',log):raise RuntimeError('Runtime model load was not confirmed')
 data.update(scene=scene,method=method,seed=seed,budget_seconds=budget,resolution=256,cpus=cpus,mse=mse(Path(out)/'image.exr',references),reference_images=[str(Path(p).resolve()) for p in references])
 (Path(out)/'result.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data),flush=True);return data
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES,required=True);p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,default=1729);p.add_argument('--seconds',type=int,required=True);p.add_argument('--cpus',required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--reference',type=Path,nargs='+',required=True);a=p.parse_args()
 run(a.scene,a.method,a.seed,a.seconds,[int(c) for c in a.cpus.split(',')],a.build,a.output,a.reference,a.model)
