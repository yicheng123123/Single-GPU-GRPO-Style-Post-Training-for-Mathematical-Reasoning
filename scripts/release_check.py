"""Inspect the prospective public snapshot; never stage, commit, push or delete."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def check():
    result=subprocess.run(['git','-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard','-z'],capture_output=True,check=True)
    files=sorted({p.decode('utf-8') for p in result.stdout.split(b'\0') if p})
    large=[];banned=[];secrets=[];private=[];broken=[];count=0
    secret_patterns={
        'hf_token':r'hf_[A-Za-z0-9]{25,}',
        'github_token':r'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',
        'api_key':r'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}',
        'aws_key':r'AKIA[A-Z0-9]{16}',
        'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    }
    private_pattern=re.compile(r'/mnt/workspace|/home/[^/\s]+/|[A-Z]:[\\/]Users[\\/][^\\/\s]+|[A-Z]:[\\/]projects|[A-Z]:[\\/]a5',re.I)
    prohibited={'.local','data','datasets','checkpoints','artifacts','.venv','.venv-wsl','__pycache__'}
    suffixes={'.pt','.pth','.ckpt','.safetensors','.bin','.parquet','.arrow','.tar','.gz','.zip'}
    for name in files:
        path=ROOT/name
        if not path.is_file():continue
        if path.stat().st_size>=100*1024*1024:large.append(name)
        if path.suffix in suffixes or set(path.parts)&prohibited:banned.append(name)
        if path.suffix in ('.png',):continue
        try:text=path.read_text(encoding='utf-8')
        except UnicodeError:continue
        count+=1
        for kind,pattern in secret_patterns.items():
            for match in re.finditer(pattern,text):secrets.append({'file':name,'line':text.count('\n',0,match.start())+1,'kind':kind})
        # The scanner's own literal detection pattern is not leaked user configuration.
        if name!='scripts/release_check.py' and private_pattern.search(text):private.append(name)
        if path.suffix=='.md':
            targets=re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)',text)+re.findall(r'<img[^>]+src="([^"]+)"',text)
            for target in targets:
                if target.startswith(('https://','http://','mailto:','#')):continue
                clean=unquote(target.split('#')[0])
                if clean and not (path.parent/clean).exists():broken.append({'file':name,'target':target})
    parent={'checked':False,'changed':[]}
    before=ROOT/'.local/a1_before_hashes.json'
    if before.exists():
        original=json.loads(before.read_text(encoding='utf-8-sig'))
        parent={'checked':True,'tracked_files':len(original),'changed':[name for name,h in original.items()
            if not (ROOT.parent/name).is_file() or digest(ROOT.parent/name)!=h]}
    return {'prospective_public_files':len(files),'text_files_scanned':count,'files_over_100MiB':large,
            'banned_artifacts':banned,'secret_pattern_hits':secrets,'private_path_hits':private,
            'broken_markdown_links':broken,'A1_integrity':parent,
            'scope':'Pattern and path checks only; not a complete security or copyright certification',
            'passed':not any((large,banned,secrets,private,broken,parent['changed']))}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');a=p.parse_args()
    r=check()
    if a.output:
        dest=Path(a.output).resolve()
        if not dest.is_relative_to(ROOT):raise ValueError('Report must stay inside project')
        if dest.exists():raise FileExistsError(dest)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False,indent=2))
    raise SystemExit(0 if r['passed'] else 1)
