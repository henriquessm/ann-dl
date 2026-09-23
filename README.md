# ann-dl — entregas de Redes Neurais e Deep Learning

Site: https://henriquessm.github.io/ann-dl/

| Exercício | Relatório | Código |
|---|---|---|
| Dados | https://henriquessm.github.io/ann-dl/exercises/data/ | `scripts/analyze.py` |
| Perceptron | https://henriquessm.github.io/ann-dl/exercises/perceptron/ | `docs/exercises/perceptron/code/perceptron.py` |

## Site (MkDocs + Material)

A partir da raiz, com as dependências de `requirements.txt` instaladas:

```sh
python -m mkdocs build --strict --site-dir _site   # páginas em docs/
python scripts/build_report.py                      # relatório de dados (formato anterior)
python scripts/build_site.py                        # copia o relatório de dados para _site/exercises/data/
python -m http.server 8000 --directory _site
```

O relatório de dados foi entregue antes da adoção do MkDocs e mantém seu próprio HTML e pipeline; o workflow em `.github/workflows/pages.yml` executa os três comandos acima e publica `_site` inteira.

## Perceptron

```sh
python docs/exercises/perceptron/code/perceptron.py
```

Gera `docs/exercises/perceptron/figures/fig1.png` a `fig6.png` e `docs/exercises/perceptron/code/results.json` com todos os números do relatório. Um único `np.random.default_rng(42)` é usado do início ao fim; execute o script inteiro.

---

# Preparação e análise de dados para redes neurais (entrega anterior)

Relatório acadêmico com três exercícios, seis figuras e código reproduzível, sem treinamento de modelos.

Site: https://henriquessm.github.io/ann-dl/exercises/data/

## Reproduzir

Com Python 3.13, a partir da raiz:

```sh
python -m pip install -r requirements.txt
python scripts/analyze.py
python scripts/build_report.py
python scripts/build_site.py
python -m http.server 8000 --directory _site
```

Abra http://localhost:8000/exercises/data/. O endereço http://localhost:8000/ redireciona para o relatório. O mesmo objeto `np.random.default_rng(42)` é usado em toda a análise, inclusive no split estratificado. Não executar trechos isolados esperando os mesmos sorteios: execute o script inteiro. As bibliotecas científicas estão fixadas em `requirements.txt`.

## Estrutura

- `index.html`: relatório, com front matter e seções na ordem da atividade.
- `_layouts/default.html` e `assets/style.css`: apresentação responsiva.
- `assets/figures/`: seis figuras solicitadas.
- `scripts/analyze.py`: geração, análise, pré-processamento e verificações.
- `scripts/build_report.py`: insere números calculados no relatório e preserva o permalink `/exercises/data/`.
- `scripts/build_site.py`: gera o relatório em `_site/exercises/data/index.html`, com os recursos em `_site/assets/`, `_site/results/` e `_site/scripts/`.
- `results/metrics.json`: todos os números e parâmetros reportados.
- `results/features.npz`: matrizes, alvos, índices originais e nomes das features.
- `results/*.csv`: tabelas descritivas.
- `data/train.csv`: arquivo analisado; origem em `data/README.md`.
- `.github/workflows/pages.yml`: publicação pelo GitHub Actions.

## GitHub Pages

1. Envie as alterações deste repositório para a branch `main`, incluindo `.github/workflows/pages.yml`.
2. Em [Settings → Pages](https://github.com/henriquessm/ann-dl/settings/pages), selecione **Source → GitHub Actions**.
3. Na aba **Actions**, acompanhe o workflow **Publicar relatório no GitHub Pages**. Se necessário, execute-o com **Run workflow** após configurar o Pages.
4. Quando a publicação terminar, abra https://henriquessm.github.io/ann-dl/exercises/data/.

O workflow regenera o relatório a partir dos resultados versionados e publica a pasta `_site` inteira a cada push em `main`. Não publique apenas `_site/exercises/data`: isso removeria o prefixo desejado e deixaria de incluir os recursos compartilhados. A pasta `_site` é saída gerada; não é necessário adicioná-la ao commit para o workflow funcionar.

O `baseurl` continua sendo `/ann-dl`. O front matter `permalink: /exercises/data/` também mantém o caminho do relatório em uma compilação com Jekyll. Os caminhos `../../assets/` e `../../results/` são links relativos no HTML; os comandos Python continuam partindo da raiz do repositório, sem `../../`.

Referência: [publicação com workflows personalizados no GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

O enunciado recebido menciona uma seção externa “Formato de Entrega”, que não foi fornecida. Foi adotada uma estrutura convencional com front matter (`layout`, `title`, `lang`); ainda é necessário confrontá-la com eventuais campos e caminhos exigidos nessa seção.

Colaboração com IA permitida pelo enunciado e declarada no relatório.
