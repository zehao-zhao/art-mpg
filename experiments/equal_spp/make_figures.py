"""Insert D01 pixels, median/SD numbers and curves into submitted strip formats."""
import argparse,csv,json,re,sys
from pathlib import Path
import numpy as np
from PIL import Image
from pdf_objects import read_pdf,stream,replace_stream,write_pdf
from strip_layout import NUMERIC,number,patch_number,compose
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from render_utils import rgb
HERE=Path(__file__).resolve().parent
ORDER=['art_mpg','npm','mpg','sms','ppg','pt'];INTERNAL=dict(zip(ORDER,['diffusion_mlp','npm_adapted','mpg_original','sms_fixed_chain_extension','ppg_style','pt']))
def display(a,exposure):
 lin=np.clip(a.astype(np.float64)*exposure,0,1);s=np.where(lin<=0.0031308,12.92*lin,1.055*np.power(lin,1/2.4)-0.055)
 return np.rint(np.clip(s,0,1)*255).astype(np.uint8)

def curves(body,old,new):
 data=stream(body)[1];changes=[];used=set()
 pattern=rb'q\s*\n((?:1 0 0 1 [^\n]+ cm /M\d+ Do\s*\n){5})Q'
 for g in re.finditer(pattern,data):
  marker=re.search(rb'/(M\d+) Do',g[1])[1];delta=np.array([[float(x),float(y)] for x,y in re.findall(rb'1 0 0 1 ([\d.eE+-]+) ([\d.eE+-]+) cm /M\d+ Do',g[1])]);xy=np.cumsum(delta,axis=0)
  candidates=[]
  for method,vals in old.items():
   slope,intercept=np.polyfit(np.log10(vals),xy[:,1],1);residual=np.max(np.abs(np.log10(vals)*slope+intercept-xy[:,1]));candidates.append((residual,method,slope,intercept))
  residual,method,slope,intercept=min(candidates)
  if residual>1e-7 or method in used:raise ValueError('Submitted curve mapping changed')
  used.add(method);newxy=np.column_stack((xy[:,0],np.log10(new[method])*slope+intercept));deltas=np.vstack((newxy[0],np.diff(newxy,axis=0)))
  changes.append((g.start(1),g.end(1),b''.join(f'1 0 0 1 {x:.10f} {y:.10f} cm /'.encode()+marker+b' Do\n' for x,y in deltas)))
  paths=list(re.finditer(rb'([\d.eE+-]+ [\d.eE+-]+ m\s*\n(?:[\d.eE+-]+ [\d.eE+-]+ l\s*\n){4})\s*S',data[:g.start()]))
  path=paths[-1];oldline=np.array([[float(x),float(y)] for x,y in re.findall(rb'([\d.eE+-]+) ([\d.eE+-]+) [ml]',path[1])])
  if np.max(np.abs(oldline-xy))>1e-5:raise ValueError('Submitted curve line mapping changed')
  changes.append((path.start(1),path.end(1),b''.join(f'{x:.6f} {y:.6f} {"m" if i==0 else "l"}\n'.encode() for i,(x,y) in enumerate(newxy))))
 if used!=set(ORDER):raise ValueError('All six curves are required')
 for start,end,replacement in sorted(changes,reverse=True):data=data[:start]+replacement+data[end:]
 return replace_stream(body,data)

def main():
 p=argparse.ArgumentParser();p.add_argument('--statistics',type=Path,required=True);p.add_argument('--crops',type=Path,required=True,help='Native D01 run directory');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 stats={(r['scene'],r['method'],int(r['SPP'])):r for r in csv.DictReader(a.statistics.open())};settings=json.loads((HERE/'display.json').read_text());old=json.loads((HERE/'format/curve_values.json').read_text());a.output.mkdir(parents=True,exist_ok=False);pages=a.output/'pages';pages.mkdir()
 crops={}
 for f in a.crops.glob('*/result.json'):
  r=json.loads(f.read_text())
  if int(r['seed'])!=1668592294 or int(r['SPP'])!=60 or not r.get('crop'):raise ValueError('Figures require only D01/60-SPP native crops')
  crops[r['scene'],r['method'],r['crop']]=f.parent/'image.exr'
 for scene in settings:
  objects,root=read_pdf(HERE/'format'/f'{scene}_strip.pdf');page=next(v for v in objects.values() if re.search(rb'/Type\s*/Page\b',v[:100]));cid=int(re.search(rb'/Contents\s+(\d+) 0 R',page)[1]);rid=int(re.search(rb'/Resources\s+(\d+) 0 R',page)[1]);resource=objects[rid];content=stream(objects[cid])[1]
  numeric=[m for m in re.finditer(rb'BT\s*\n(.*?)\nET',content,re.S) if re.search(NUMERIC,m[1],re.S)]
  if len(numeric)!=12:raise ValueError('Submitted numeric layout changed')
  for i,method in enumerate(ORDER):
   for j,field in enumerate(['median','sample_SD']):
    b=numeric[2*i+j][1];value=float(stats[scene,INTERNAL[method],60][field]);content=content.replace(b,patch_number(b,value),1)
   for crop,index in [('crop1',3+2*i),('crop2',4+2*i)]:
    image_id=int(re.search(f'/Im{index}\\s+(\\d+) 0 R'.encode(),resource)[1]);arr=rgb(crops[scene,INTERNAL[method],crop])
    if arr.shape!=(288,288,3):raise ValueError('Native crop must be 288×288')
    pixels=display(arr,settings[scene]['exposure']);objects[image_id]=replace_stream(objects[image_id],b''.join(b'\0'+row.tobytes() for row in pixels))
  objects[cid]=replace_stream(objects[cid],content);curve_id=int(re.search(rb'/Im2\s+(\d+) 0 R',resource)[1]);new={m:[float(stats[scene,INTERNAL[m],s]['median']) for s in [12,24,36,48,60]] for m in ORDER};objects[curve_id]=curves(objects[curve_id],old[scene],new);write_pdf(pages/f'{scene}_strip.pdf',objects,root)
 for name in ['figure1.tex','appendix_figures.tex']:(a.output/name).write_text((HERE/'format'/name).read_text())
 compose([pages/(s+'_strip.pdf') for s in ['living','water','stone','pool']],a.output/'Figure1.pdf',gap=.5*72/72.27)
 compose([pages/(s+'_strip.pdf') for s in ['sphere','lamp','plane','slabs','ring_and_slab','pillars','slab_x1','slab_x3','pillar_x3','cornell_glass']],a.output/'Appendix.pdf',gap=.5*72/72.27)
 print('Generated submitted-format scene strips, Figure 1 and appendix inclusions')
if __name__=='__main__':main()
