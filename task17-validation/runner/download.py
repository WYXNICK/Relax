import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

root=Path('/tmp/relax-task17')
models=[('Qwen/Qwen-Image-Edit','ac7f9318f633fc4b5778c59367c8128225f1e3de'),('sensenova/SenseNova-U1.5-8B-MoT','9feeeab8a2792514d109cd34589342a2cc1d4ab2')]
def download(pair):
    repo,rev=pair
    directory=root/'models'/repo.split('/')[-1]
    print('Downloading',repo,rev,flush=True)
    snapshot_download(repo,revision=rev,local_dir=directory,max_workers=4)
    files=[]
    for path in sorted(directory.rglob('*')):
        if path.is_file() and '.cache' not in path.parts:
            digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
            files.append({'path':str(path.relative_to(directory)),'bytes':path.stat().st_size,'sha256':digest})
    (root/'models'/f'{directory.name}.manifest.json').write_text(json.dumps({'repo':repo,'revision':rev,'files':files},indent=2))
    print('Complete',repo,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(download,models))
