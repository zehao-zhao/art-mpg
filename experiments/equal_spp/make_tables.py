"""Compute linear RGB MSE medians, means and sample SD; fill submitted tables."""
import argparse,csv,json,re,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from render_utils import mse
HERE=Path(__file__).resolve().parent
SCENES=['living','lamp','plane','slabs','stone','pool','sphere','water','ring_and_slab','pillars','slab_x1','slab_x3','pillar_x3','cornell_glass'];METHODS=['pt','ppg_style','sms_fixed_chain_extension','mpg_original','npm_adapted','diffusion_mlp']
SEEDS={r['render_seed'] for r in json.loads((HERE.parent/'recorded_seeds.json').read_text())['paper']}
NUM_TEX=r'([0-9]+\.[0-9]+)((?:\\!)*\\times(?:\\!)*10\^\{)(-?[0-9]+)(\})'
def replace_values(text,values):
 matches=list(re.finditer(NUM_TEX,text))
 if len(matches)!=len(values):raise ValueError('Table format/values count mismatch')
 for m,value in reversed(list(zip(matches,values))):
  mant,exp=f'{value:.3e}'.split('e');text=text[:m.start()]+mant+m[2]+str(int(exp))+m[4]+text[m.end():]
 return text

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--references',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.runs.is_file():rows=list(csv.DictReader(a.runs.open()))
 else:
  refs=json.loads(a.references.read_text()) if a.references else {};rows=[]
  for f in a.runs.glob('*/result.json'):
   r=json.loads(f.read_text())
   if r.get('crop'):continue
   r['mse']=r.get('mse') if r.get('mse') is not None else mse(f.parent/'image.exr',[refs[r['scene']]])
   rows.append(r)
 groups={}
 for r in rows:
  key=(r['scene'],r.get('internal_method',r.get('method')),int(r['SPP']));g=groups.setdefault(key,{})
  seed=int(r['seed'])
  if seed in g:raise ValueError(f'Duplicate {key}/{seed}')
  g[seed]=float(r['mse'] if 'mse' in r else r['recomputed_MSE'])
 stats=[]
 for scene in SCENES:
  for method in METHODS:
   for spp in [12,24,36,48,60]:
    g=groups.get((scene,method,spp),{})
    if set(g)!=SEEDS:raise ValueError(f'Incomplete cell: {scene}/{method}/{spp}')
    vals=list(g.values());stats.append(dict(scene=scene,method=method,SPP=spp,n=10,median=statistics.median(vals),mean=statistics.mean(vals),sample_SD=statistics.stdev(vals)))
 a.output.mkdir(parents=True,exist_ok=False)
 with (a.output/'statistics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
 index={(r['scene'],r['method'],r['SPP']):r for r in stats}
 for file,fields in [('table1.tex',['median']),('appendix_table.tex',['median','mean','sample_SD'])]:
  values=[index[s,m,60][field] for s in SCENES for m in METHODS for field in fields]
  (a.output/file).write_text(replace_values((HERE/'format'/file).read_text(),values))
 print('Generated 420 ten-seed statistic cells and both submitted table templates')
if __name__=='__main__':main()
