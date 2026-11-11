"""Copy Python and available dependencies without source atime writes.

Only for the observed 9207 Btrfs ENOSPC incident. Uses rsync --open-noatime;
does not rebalance, remount, delete, or change the source installation.
"""
import json
from pathlib import Path
import subprocess


SOURCE=Path('/home/wuzewen/miniconda3')
ROOT=Path('/dev/shm/edgecloud-stage-a-wuzewen-01a15efc')
DEST=ROOT/'runtime'


def copy(sources,destination,extra=()):
    destination.mkdir(parents=True,exist_ok=True)
    print('copying '+', '.join(str(p) for p in sources),flush=True)
    subprocess.run(['rsync','-aL','--open-noatime',*extra,
                    *map(str,sources),str(destination)+'/'],check=True)


def main():
    copy([SOURCE/'bin/python3.13'],DEST/'bin')
    copy([str(SOURCE/'lib/python3.13')+'/'],DEST/'lib/python3.13',extra=('--exclude=site-packages',))
    for name in ['libssl.so.3','libcrypto.so.3','libffi.so.8','libbz2.so.1.0','liblzma.so.5','libz.so.1']:
        source=SOURCE/'lib'/name
        if source.exists():copy([source],DEST/'lib')
    site=SOURCE/'lib/python3.13/site-packages'
    wanted=['pip','torch','torchgen','functorch','nvidia','numpy','numpy.libs','networkx',
            'sympy','mpmath','jinja2','markupsafe','fsspec','filelock','typing_extensions.py',
            'packaging','yaml','_yaml','regex','requests','charset_normalizer',
            'charset_normalizer.libs','idna','urllib3','certifi','tqdm']
    for name in wanted:
        if (site/name).exists():copy([site/name],DEST/'lib/python3.13/site-packages')
    prefixes=tuple(name.replace('.py','').replace('.','_').lower()+'-' for name in wanted)+('pyyaml-',)
    for path in site.iterdir():
        if path.name.endswith('.dist-info') and (path.name.lower().startswith(prefixes) or path.name.lower().startswith('nvidia_')):
            copy([path],DEST/'lib/python3.13/site-packages')
    print(json.dumps({'status':'python_runtime_copy_completed','runtime':str(DEST),
                      'torch_directory_copied':(DEST/'lib/python3.13/site-packages/torch').is_dir()}),flush=True)


if __name__=='__main__':main()
