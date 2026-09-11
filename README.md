# Preparação e análise de dados para redes neurais

Relatório acadêmico com três exercícios, seis figuras e código reproduzível, sem treinamento de modelos.

Site previsto: https://henriquessm.github.io/ann-dl/

## Reproduzir

Com Python 3.13, a partir da raiz:

```sh
python -m pip install -r requirements.txt
python ../../scripts/analyze.py
python ../../scripts/build_report.py
python ../../scripts/build_site.py
python -m http.server 8000 --directory _site
```

Abra http://localhost:8000. O mesmo objeto `np.random.default_rng(42)` é usado em toda a análise, inclusive no split estratificado. Não executar trechos isolados esperando os mesmos sorteios: execute o script inteiro. As bibliotecas científicas estão fixadas em `requirements.txt`.

## Estrutura

- `index.html`: relatório, com front matter e seções na ordem da atividade.
- `_layouts/default.html` e `../../assets/style.css`: apresentação responsiva.
- `../../assets/figures/`: seis figuras solicitadas.
- `../../scripts/analyze.py`: geração, análise, pré-processamento e verificações.
- `../../scripts/build_report.py`: insere números calculados no relatório.
- `../../scripts/build_site.py`: gera `_site/` para publicação estática.
- `../../results/metrics.json`: todos os números e parâmetros reportados.
- `../../results/features.npz`: matrizes, alvos, índices originais e nomes das features.
- `../../results/*.csv`: tabelas descritivas.
- `data/train.csv`: arquivo analisado; origem em `data/README.md`.
- `.github/workflows/pages.yml`: publicação pelo GitHub Actions.

## GitHub Pages

Em Settings → Pages, use Source → GitHub Actions. O workflow publica a versão calculada e verificada do relatório a cada push em `main`. A primeira ativação de Pages pode exigir uma pessoa com permissão de administrador; se o passo `configure-pages` falhar, ative a fonte GitHub Actions em Settings e execute o workflow novamente.

O enunciado recebido menciona uma seção externa “Formato de Entrega”, que não foi fornecida. Foi adotada uma estrutura convencional com front matter (`layout`, `title`, `lang`); ainda é necessário confrontá-la com eventuais campos e caminhos exigidos nessa seção.

Colaboração com IA permitida pelo enunciado e declarada no relatório.
