"""Five-seed median MSE/SPP tables and offline training clocks."""
import argparse,csv,json,statistics,math
from pathlib import Path
SCENES=['living','slabs','cornell_glass','pool','water','stone'];METHODS=['art_joint','mpg','npm'];SEEDS={1729,31415,271828,424242,8675309}
NAMES={'art_joint':'ART-MPG (joint R/T, cache 0.25,4096)','mpg':'MPG','npm':'NPM'}
ALIASES={'Joint R/T (local cache0.25,4096slots)':'art_joint','MPG':'mpg','NPM-adapted':'npm'}
def write_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def aggregate(rows):
 groups={}
 for r in rows:
  method=ALIASES.get(r['method'],r['method']);budget=int(r['budget_seconds']);scene=r['scene']
  if method not in METHODS or scene not in SCENES or budget not in (180,300):continue
  seed=int(r['seed']);key=(scene,budget,method);g=groups.setdefault(key,{})
  if seed in g:raise ValueError(f'Duplicate seed: {key}, {seed}')
  error=float(r['mse']);spp=int(r['spp_reached'])
  if not math.isfinite(error) or error<0 or spp<=0:raise ValueError('Invalid MSE/SPP')
  g[seed]=(error,spp)
 result=[]
 for scene in SCENES:
  for budget in (180,300):
   for method in METHODS:
    key=(scene,budget,method);g=groups.get(key,{})
    if set(g)!=SEEDS:raise ValueError(f'Incomplete five-seed cell {key}: {sorted(g)}')
    result.append(dict(scene=scene,budget_seconds=budget,method=method,runs=5,seeds_used=' '.join(map(str,sorted(SEEDS))),median_mse=statistics.median(v[0] for v in g.values()),median_spp=statistics.median(v[1] for v in g.values())))
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True,help='Per-run CSV or directory of result.json files');p.add_argument('--training',type=Path,required=True,help='CSV: scene,method,training_seconds,clock_scope,threads');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rows=list(csv.DictReader(a.runs.open())) if a.runs.is_file() else [json.loads(f.read_text()) for f in a.runs.glob('*/result.json')]
 med=aggregate(rows);training=list(csv.DictReader(a.training.open()));clocks={}
 for r in training:
  method={'npm_adapted':'npm','joint_rt':'art_joint'}.get(r['method'],r['method'])
  if r['scene'] in SCENES and method in ('art_joint','npm'):
   key=(r['scene'],method)
   if key in clocks:raise ValueError(f'Duplicate training clock {key}')
   sec=float(r['training_seconds'])
   if not math.isfinite(sec) or sec<=0:raise ValueError('Invalid training seconds')
   clocks[key]=dict(scene=r['scene'],method=method,training_seconds=sec,clock_scope=r['clock_scope'],threads=int(r['threads']))
 if len(clocks)!=12:raise ValueError('Need both offline training clocks for all six scenes')
 a.output.mkdir(parents=True,exist_ok=False);write_csv(a.output/'medians.csv',med);write_csv(a.output/'training_times.csv',[clocks[s,m] for s in SCENES for m in ('art_joint','npm')])
 lookup={(r['scene'],r['budget_seconds'],r['method']):r for r in med}
 for name,field,title in [('Table_R1','median_mse','Median linear RGB MSE'),('Table_R2','median_spp','Median final-film SPP')]:
  lines=[f'# {name}: {title}','','256×256, 16 physical cores, five recorded seeds per cell.','', '| Scene | Budget | ART-MPG joint R/T | MPG | NPM |','|---|---:|---:|---:|---:|']
  tex=['\\begin{tabular}{llrrr}','\\toprule','Scene & Budget & ART-MPG joint R/T & MPG & NPM \\\\','\\midrule']
  for scene in SCENES:
   for budget in (180,300):
    vals=[lookup[scene,budget,m][field] for m in METHODS];fmt=[f'{v:.6g}' if field=='median_mse' else str(v) for v in vals]
    lines.append('| '+ ' | '.join([scene,str(budget)+' s',*fmt])+' |');tex.append(scene.replace('_',r'\_')+' & '+str(budget)+' s & '+' & '.join(fmt)+r' \\')
  tex+=['\\bottomrule','\\end{tabular}']
  if name=='Table_R2':
   lines+=['','## Offline training time','','| Scene | ART-MPG joint s | NPM s |','|---|---:|---:|'];tt=['\\begin{tabular}{lrr}','\\toprule','Scene & ART-MPG joint s & NPM s \\\\','\\midrule']
   for scene in SCENES:
    vals=[clocks[scene,m]['training_seconds'] for m in ('art_joint','npm')];lines.append(f'| {scene} | {vals[0]:.6g} | {vals[1]:.6g} |');tt.append(scene.replace('_',r'\_')+' & '+' & '.join(f'{v:.6g}' for v in vals)+r' \\')
   tt+=['\\bottomrule','\\end{tabular}'];tex+=['',*tt]
   scopes=sorted({f"{NAMES[r['method']]}: {r['clock_scope']} ({r['threads']} cores)" for r in clocks.values()});lines+=['',*scopes,'','Offline clocks are separate from render budgets; replay acquisition is excluded.'];tex+=['',r'\par\footnotesize ART joint: training, validation, checkpoint and runtime export; NPM: training process wall clock, runtime export excluded. Data acquisition excluded.']
  (a.output/(name+'.md')).write_text('\n'.join(lines)+'\n');(a.output/(name+'.tex')).write_text('\n'.join(tex)+'\n')
 print('Generated 36 five-seed cells and 12 offline clocks')
if __name__=='__main__':main()
