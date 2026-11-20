"""Compile the three R1/R2 kernels against a freshly built package renderer."""
import argparse,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
METHODS=['art_joint','mpg','npm']
def main():
 p=argparse.ArgumentParser();p.add_argument('--base-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 dist=a.base_build.resolve()/'dist';src=ROOT/'512diffusion/mitsuba'
 if not (dist/'mitsuba').is_file():raise FileNotFoundError('Build the package renderer first')
 flags=['-DLITTLE_ENDIAN','-DMTS_ENABLE_PROFILER','-D__LINUX__','-std=gnu++17','-O3','-DNDEBUG','-fPIC','-fno-stack-protector','-fomit-frame-pointer','-fno-math-errno','-ffp-contract=fast','-march=native','-fopenmp','-Wall','-Wextra','-Wno-unused-local-typedefs','-fpermissive','-finline-limit=150','-Wdouble-promotion']
 includes=[src/'include',src/'ext/tinyformat',src/'ext/enoki/include',src/'ext/tbb/include',Path('/usr/include/libcxxabi')]
 a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
 for method in METHODS:
  target=a.output/method;target.mkdir();shutil.copytree(dist,target/'dist',symlinks=False)
  native=Path(__file__).resolve().parent/'native'/method/'MPG/manifold_path_guiding.cpp'
  subprocess.run(['c++',*flags,*['-I'+str(i) for i in includes],'-c',str(native),'-o',str(target/'kernel.o')],check=True)
  subprocess.run(['c++','-shared','-march=native','-fopenmp',str(target/'kernel.o'),'-o',str(target/'dist/plugins/manifold_path_guiding.so'),'-L'+str(target/'dist'),'-lmitsuba-render','-lmitsuba-core','-ltbb','-lpthread','-ldl','-Wl,-rpath,$ORIGIN/..'],check=True)
  print('Built',method,flush=True)
if __name__=='__main__':main()
