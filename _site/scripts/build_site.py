"""Renderização estática do layout e front matter, sem dependências adicionais."""
from pathlib import Path
import html
import shutil
ROOT=Path(__file__).resolve().parents[1]
DEST = ROOT / '_site'
REPORT_DEST = DEST / 'exercises' / 'data'
REPORT_DEST.mkdir(parents=True, exist_ok=True)
source=(ROOT/'index.html').read_text(encoding='utf-8')
assert source.startswith('---\n')
_,front,content=source.split('---',2)
title=next(line.split(':',1)[1].strip().strip('"') for line in front.splitlines() if line.startswith('title:'))
layout=(ROOT/'_layouts/default.html').read_text(encoding='utf-8')
rendered=layout.replace('{{ page.title }}',html.escape(title)).replace('{{ content }}',content)
(REPORT_DEST/'index.html').write_text(rendered,encoding='utf-8',newline='\n')
# Os links ../../assets, ../../results e ../../scripts partem do relatório.
# Esses arquivos ficam na raiz do site, como na publicação com Jekyll.
for folder in ['assets','results','scripts']:
    shutil.copytree(ROOT/folder,DEST/folder,dirs_exist_ok=True)
shutil.copy2(ROOT/'requirements.txt',DEST/'requirements.txt')
(DEST/'index.html').write_text('''<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="0; url=exercises/data/">
  <title>Relatório de análise de dados</title>
</head>
<body><p><a href="exercises/data/">Abrir o relatório de análise de dados</a></p></body>
</html>
''', encoding='utf-8', newline='\n')
(DEST/'.nojekyll').write_text('', encoding='utf-8')
print('Site pronto em _site/exercises/data/index.html')
