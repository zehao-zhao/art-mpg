"""Train/export the two offline R1/R2 models and record their stated clocks."""
import argparse,csv,json,os,subprocess,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from render_utils import ROOT,CoreLease
SCENES=['living','slabs','cornell_glass','pool','water','stone'];HERE=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES);p.add_argument('--data',type=Path,required=True,help='Directory containing six exact replay files scene.jsonl');p.add_argument('--output',type=Path,required=True);p.add_argument('--joint-cpu',type=int,required=True);p.add_argument('--npm-cpus',required=True);a=p.parse_args()
 a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);clocks=[];models={};spec=json.loads((HERE/'training_data.json').read_text());npm_cpus=list(map(int,a.npm_cpus.split(',')))
 if len(npm_cpus)!=4:raise ValueError('The NPM training clocks use four physical cores')
 for scene in ([a.scene] if a.scene else SCENES):
  replay=a.data.resolve()/(scene+'.jsonl')
  if hashlib.sha256(replay.read_bytes()).hexdigest()!=spec[scene]['sha256']:raise ValueError('Use the specified complete replay for '+scene)
  joint=a.output/scene/'art_joint';joint.parent.mkdir()
  with CoreLease([a.joint_cpu]):subprocess.run(['taskset','-c',str(a.joint_cpu),sys.executable,'-B',str(HERE/'train_joint_rt.py'),'--scene',scene,'--replay',str(replay),'--output',str(joint)],check=True)
  j=json.loads((joint/'training.json').read_text());clocks.append({k:j[k] for k in ['scene','method','training_seconds','clock_scope','threads']});models[scene]={'art_joint':str(joint/'runtime.txt')}
  config=json.loads((ROOT/'models/configs'/scene/'npm_adapted_seed104729.json').read_text());config['replay_paths']=[str(replay)];config['output_dir']=str(a.output/scene/'npm');config_path=a.output/scene/'npm_config.json';config_path.write_text(json.dumps(config,indent=2)+'\n')
  env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
  begin=time.perf_counter()
  with CoreLease(npm_cpus):subprocess.run(['taskset','-c',a.npm_cpus,sys.executable,'-B',str(ROOT/'models/scripts/train_npm.py'),'--config',str(config_path)],check=True,env=env)
  seconds=time.perf_counter()-begin;clocks.append(dict(scene=scene,method='npm',training_seconds=seconds,clock_scope='training process wall clock; runtime export excluded',threads=4))
  checkpoints=list((a.output/scene/'npm').glob('*/npm_adapted_seed_direction.pt'))
  if len(checkpoints)!=1:raise RuntimeError('NPM checkpoint count is not one')
  runtime=a.output/scene/'npm_runtime';export=dict(checkpoint_path=str(checkpoints[0]),model_kind='npm_adapted',output_dir=str(runtime),output_name='runtime.txt');exp=a.output/scene/'npm_export_config.json';exp.write_text(json.dumps(export,indent=2)+'\n')
  subprocess.run([sys.executable,'-B',str(ROOT/'models/scripts/export_art_mpg.py'),'--config',str(exp)],check=True,env=env);models[scene]['npm']=str(runtime/'runtime.txt')
 with (a.output/'training_times.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(clocks[0]));w.writeheader();w.writerows(clocks)
 (a.output/'models.json').write_text(json.dumps(models,indent=2)+'\n')
