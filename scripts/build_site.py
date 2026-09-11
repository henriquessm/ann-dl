"""Renderização estática do layout e front matter, sem dependências adicionais."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'_site'
DEST.mkdir(exist_ok=True)
source=(ROOT/'index.html').read_text(encoding='utf-8')
assert source.startswith('---\n')
_,front,content=source.split('---',2)
title=next(line.split(':',1)[1].strip().strip('"') for line in front.splitlines() if line.startswith('title:'))
layout=(ROOT/'_layouts/default.html').read_text(encoding='utf-8')
rendered=layout.replace('{{ page.title }}',title).replace('{{ content }}',content)
(DEST/'index.html').write_text(rendered,encoding='utf-8')
for folder in ['assets','results','scripts']:
    shutil.copytree(ROOT/folder,DEST/folder,dirs_exist_ok=True)
shutil.copy2(ROOT/'requirements.txt',DEST/'requirements.txt')
print('Site pronto em _site/index.html')
