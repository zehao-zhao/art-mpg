"""Scene preparation, physical-core leases and linear RGB error evaluation."""
import os,re,json,subprocess,time,fcntl
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
def rgb(path):
 import cv2
 a=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
 if a is None:raise ValueError(f'Cannot decode {path}')
 if a.ndim==2:a=np.repeat(a[:,:,None],3,axis=2)
 a=a[:,:,:3][:,:,::-1].copy().astype(np.float32)
 if not np.isfinite(a).all():raise ValueError(f'Nonfinite RGB: {path}')
 return a

def mse(image,references):
 a=rgb(image).astype(np.float64);ref=np.mean([rgb(p).astype(np.float64) for p in references],axis=0)
 if a.shape!=ref.shape:raise ValueError(f'Image/reference shape mismatch: {a.shape}, {ref.shape}')
 return float(np.mean((a-ref)**2))

def scalar(text,name,value):
 return re.sub(r'((?:name="'+re.escape(name)+r'"\s+value="))[^"]*',lambda m:m[1]+str(value),text)

def prepare(text,asset_root,resolution,spp,seed,budget=None,model=None,roi=None):
 text=text.replace('${ARTMPG_ROOT}',str(ROOT))
 defaults=dict(re.findall(r'<default\s+name="([^"]+)"\s+value="([^"]+)"',text))
 def assets(m):
  val=re.sub(r'\$(\w+)',lambda n:defaults.get(n[1],n[0]),m[2]);p=Path(val)
  if not p.is_absolute():p=asset_root/p
  if not p.is_file():raise FileNotFoundError(p)
  return m[1]+str(p.resolve())+m[3]
 text=re.sub(r'(<string\s+name="filename"\s+value=")([^"]+)(")',assets,text)
 text=re.sub(r'(<include\s+filename=")([^"]+)(")',assets,text)
 props=dict(width=resolution,height=resolution,imgwidth=resolution,imgheight=resolution,crop_offset_x=0,crop_offset_y=0,crop_width=resolution,crop_height=resolution,spp=spp,sample_count=spp,seedid=seed,seed=seed,timeout=budget or 0,budget_is_time='true' if budget else 'false')
 if roi:props.update(dict(zip(['crop_offset_x','crop_offset_y','crop_width','crop_height'],roi)))
 for k,v in props.items():text=scalar(text,k,v)
 # Film crop fields are optional in original 64-pixel source scenes.
 import xml.etree.ElementTree as ET
 root=ET.fromstring(text);film=root.find('.//film');sampler=root.find('.//sampler')
 for k in ['width','height','crop_offset_x','crop_offset_y','crop_width','crop_height']:
  e=film.find(f"integer[@name='{k}']")
  if e is None:e=ET.SubElement(film,'integer',name=k)
  e.set('value',str(props[k]))
 e=sampler.find("integer[@name='seed']")
 if e is None:e=ET.SubElement(sampler,'integer',name='seed')
 e.set('value',str(seed))
 if model:
  p=Path(model);p=p if p.is_absolute() else ROOT/p
  if not p.is_file():raise FileNotFoundError(f'Model export required: {p}')
  root.find(".//string[@name='masked_diffusion_runtime_path']").set('value',str(p.resolve()))
 return ET.tostring(root,encoding='unicode')

class CoreLease:
 """Locks physical cores, including SMT siblings, across package launchers."""
 def __init__(self,cpus):self.cpus=cpus;self.files=[]
 def __enter__(self):
  keys=[]
  for cpu in self.cpus:
   d=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology');keys.append(((d/'physical_package_id').read_text().strip(),(d/'core_id').read_text().strip()))
  if len(set(keys))!=len(keys):raise ValueError('SMT sharing within a render is forbidden')
  try:
   for pkg,core in sorted(keys):
    f=open(f'/tmp/artmpg-core-{os.getuid()}-{pkg}-{core}.lock','a');self.files.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BaseException:self.__exit__(None,None,None);raise
  return self
 def __exit__(self,*args):
  for f in self.files:f.close()
  self.files=[]

def render(binary,xml,out,cpus,budget=None):
 binary=Path(binary).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);(out/'scene.xml').write_text(xml)
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(binary.parent)+':'+env.get('LD_LIBRARY_PATH',''))
 cmd=['taskset','-c',','.join(map(str,cpus)),str(binary),'-m','scalar_rgb','-t',str(len(cpus)),'-o',str(out/'image.exr'),str(out/'scene.xml')]
 begin=time.perf_counter()
 with CoreLease(cpus),(out/'render.log').open('w') as log:
  p=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=log,timeout=(budget+600 if budget else 86400))
 elapsed=time.perf_counter()-begin;log=(out/'render.log').read_text()
 completed=[int(s) for s in re.findall(r'Rendering finished\. Computed (\d+) spp',log)]
 if p.returncode or not (out/'image.exr').is_file():raise RuntimeError(f'Render failed; see {out}/render.log')
 a=rgb(out/'image.exr')
 if not completed:
  completed=[int(s) for s in re.findall(r'Starting render job \(\d+x\d+, (\d+) sample',log)]
 if not completed:raise RuntimeError('Renderer did not report completed SPP')
 return dict(spp_reached=completed[-1],process_seconds=elapsed,shape=list(a.shape),finite=True,returncode=p.returncode),log
