"""Paper quantitative renders and native D01 Figure 1/appendix crops."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from render_utils import ROOT,prepare,render,mse
HERE=Path(__file__).resolve().parent

def run(scene,method,seed,model_seed,spp,binary,out,cpu,reference,crop=None):
 xml=(HERE/'scenes'/f'{scene}__{method}.xml').read_text();mapping=json.loads((HERE/'scene_paths.json').read_text());models=json.loads((HERE/'models.json').read_text())
 if method in ('diffusion_mlp','npm_adapted'):
  model=models[f'{scene}|{method}|{model_seed}'];xml=xml.replace('${MODEL_EXPORT}','${ARTMPG_ROOT}/'+model)
 else:model=None
 display=json.loads((HERE/'display.json').read_text());roi=display[scene]['rois'][crop] if crop else None
 xml=prepare(xml,ROOT/mapping[scene],1152 if crop else 64,spp,seed,None,model,roi)
 data,log=render(binary,xml,out,[cpu]);data.update(scene=scene,method=method,seed=seed,model_seed=model_seed,SPP=spp,crop=crop)
 if not crop:data['mse']=mse(Path(out)/'image.exr',[reference])
 (Path(out)/'result.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--method',required=True);p.add_argument('--seed',type=int,default=1668592294);p.add_argument('--model-seed',type=int,default=104729);p.add_argument('--spp',type=int,default=60);p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu',type=int,required=True);p.add_argument('--reference',type=Path);p.add_argument('--crop',choices=['crop1','crop2']);a=p.parse_args()
 if not a.crop and not a.reference:p.error('Quantitative MSE requires --reference')
 run(a.scene,a.method,a.seed,a.model_seed,a.spp,a.binary,a.output,a.cpu,a.reference,a.crop)
