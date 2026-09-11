"""Monta o relatório a partir dos números efetivamente calculados."""
from pathlib import Path
import json
import html
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT/'results/metrics.json').read_text(encoding='utf-8'))
r = m['real']
def num(x, digits=4):
    return f'{x:,.{digits}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
def pct(x): return num(x*100,2)+'%'
def table(headers, rows):
    head=''.join(f'<th scope="col">{html.escape(str(v))}</th>' for v in headers)
    body=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)
    return f'<div class="table-scroll" tabindex="0" role="region" aria-label="Tabela de resultados"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
def figure(n, caption):
    return f'<figure id="figura-{n}"><a href="assets/figures/figura-{n}.png"><img src="assets/figures/figura-{n}.png" alt="{html.escape(caption)}" loading="lazy"></a><figcaption><strong>Figura {n}.</strong> {caption} Clique para ampliar.</figcaption></figure>'
code=(ROOT/'scripts/analyze.py').read_text(encoding='utf-8')
cut1=code.index('# Exercício 2 A e B')
cut2=code.index('# Exercício 3:')
def codeblock(text, label):
    return f'<details><summary>{label}</summary><pre><code>{html.escape(text)}</code></pre></details>'
missing=pd.read_csv(ROOT/'results/faltantes.csv')
globalstats=pd.read_csv(ROOT/'results/gastos-global.csv',index_col=0)
trainstats=pd.read_csv(ROOT/'results/gastos-treino.csv',index_col=0)
separation=table(['Par de classes','rᵢⱼ em s = 1'],[[p['par'],num(p['r'])] for p in m['separation']])
mix=table(['s','Pontos misturados / 400','Taxa de mistura','Pares com fechos convexos sobrepostos'],
    [[s,f'{round(rate*400)} / 400',pct(rate),', '.join(m['overlap_pairs'][s]) or 'Nenhum'] for s,rate in m['mixture'].items()])
missingtable=table(['Coluna','Faltantes','Percentual'],[[row.Coluna,row.Faltantes,num(row.Percentual,2)+'%'] for row in missing.itertuples()])
def stats_table(frame): return table(['Coluna','Média','Mediana','Máximo'],[[c,*[num(v,2) for v in row]] for c,row in frame.iterrows()])
pca_table=table(['Dataset','Distância entre centros amostrais (5D)','PC1','PC2','PC1 + PC2'],
    [[label,num(m[key]['center_distance']),pct(m[key]['explained_variance'][0]),pct(m[key]['explained_variance'][1]),pct(sum(m[key]['explained_variance']))] for label,key in [('I','dataset_1'),('II','dataset_2')]])
ranges=table(['Feature numérica','Mín. treino','Máx. treino','Mín. teste','Máx. teste'],
    [[k,*[num(v,4) for v in vals.values()]] for k,vals in r['numeric_ranges'].items()])
summary_rows=[
    ['Taxa de mistura em s = 0,5',pct(m['mixture']['0.5'])],
    ['Taxa de mistura em s = 1,0',pct(m['mixture']['1.0'])],
    ['Taxa de mistura em s = 2,0',pct(m['mixture']['2.0'])],
    ['Taxa de mistura em s = 4,0',pct(m['mixture']['4.0'])],
    ['Menor rᵢⱼ em s = 1,0 e par',num(m['separation'][0]['r'])+'; par 0–1'],
    ['Distância entre os centros — Dataset I',num(m['dataset_1']['center_distance'])],
    ['Distância entre os centros — Dataset II',num(m['dataset_2']['center_distance'])],
    ['Variância explicada PC1 + PC2 — Dataset I',pct(sum(m['dataset_1']['explained_variance']))],
    ['Variância explicada PC1 + PC2 — Dataset II',pct(sum(m['dataset_2']['explained_variance']))],
    ['Proporção positiva em Transported',pct(r['positive_fraction'])+' (4.378 / 8.693)'],
    ['Média e mediana de FoodCourt no treino, antes da transformação',num(r['foodcourt_train']['mean'],4)+' e '+num(r['foodcourt_train']['median'],2)],
    ['Shape final das features de treino',str(tuple(r['shape_train']))],
    ['Mínimo e máximo após escalonamento','Treino: −1,0000 e 1,0000; teste: −1,0000 e 1,0000'],
]
body=f'''
<header class="hero">
<p class="course">Redes neurais · Análise de dados</p>
<h1>Preparação e análise de dados para redes neurais</h1>
<p class="lead">O espalhamento dos dados conecta três problemas: separar nuvens, reconhecer estruturas radiais e preparar entradas para a ativação tanh.</p>
<p>Relatório computacional reproduzível. Uma única semente: <code>rng = np.random.default_rng(42)</code>. Nenhum modelo foi treinado.</p>
<div class="downloads"><a href="scripts/analyze.py" download>Baixar código Python</a><a href="results/metrics.json">Consultar números completos</a></div>
</header>
<section id="exercicio-1">
<h2>Exercício 1 — Nuvens de pontos</h2>
<p>Abordagem: gerar gaussianas bidimensionais, aumentar seus desvios padrão e comparar uma medida de mistura com a separabilidade geométrica das amostras.</p>
<h3>A. Geração das quatro classes</h3>
<p>Foram geradas 400 amostras, 100 por classe. As coordenadas x e y são independentes dentro de cada classe; a covariância é diagonal, com as variâncias iguais aos quadrados dos desvios indicados.</p>
{table(['Classe','Média μ','Desvio padrão σ','Amostras'],[['0','[2, 3]','[0,8; 2,5]',100],['1','[5, 6]','[1,2; 1,9]',100],['2','[8, 1]','[0,9; 0,9]',100],['3','[15, 4]','[0,5; 2,0]',100]])}
{figure(1,'Dataset original, s = 1. As marcas X indicam as médias geradoras, não as médias amostrais. As linhas tracejadas são um esboço de fronteiras curvas usando os parâmetros conhecidos.')}
<h3>B. Espalhamento, separação e mistura</h3>
<p>Além do dataset original, foram gerados quatro novos datasets, cada um com 400 pontos e as mesmas quatro classes. A sequência de sorteios é: original, s = 0,5, s = 1,0, s = 2,0 e s = 4,0. São realizações independentes: o painel s = 1 da Figura 2 não repete os pontos da Figura 1. Isso mantém o mesmo gerador aleatório e torna explícita a pequena variação amostral entre painéis.</p>
{figure(2,'Quatro datasets com médias fixas e desvios multiplicados por s. Todos os painéis compartilham os mesmos limites de ambos os eixos.')}
<p>A razão de separação usa os parâmetros da distribuição, não estimativas amostrais:</p>
<div class="equation">rᵢⱼ = ‖μᵢ − μⱼ‖ / (σ̄ᵢ + σ̄ⱼ), &nbsp; σ̄ₖ = (σₖ,ₓ + σₖ,ᵧ) / 2.</div>
{separation}
<p>O menor valor é <strong>{num(m['separation'][0]['r'])}, no par 0–1</strong>. Como rᵢⱼ(s) = rᵢⱼ(1)/s, em s = 2 esse mínimo cai para <strong>{num(m['separation'][0]['r']/2)}</strong>, sem necessidade de gerar novos dados. A média dos desvios resume as direções de dispersão, mas perde informação de anisotropia; portanto, r não é um teste de separabilidade.</p>
<p>A taxa de mistura é a fração de pontos cuja média geradora mais próxima, pela distância euclidiana, pertence a outra classe. São apenas distâncias às quatro médias, calculadas com NumPy.</p>
{mix}
{figure(3,'Taxas de mistura: 0,00%, 6,75%, 22,50% e 41,75% nas quatro escalas. A linha conecta as medições; não representa um modelo ajustado.')}
<p><strong>Entre as escalas avaliadas, a primeira perda de separação estrita por retas ocorre em s = 1.</strong> Nesse ponto, o menor r é {num(m['separation'][0]['r'])}; ele diminui de {num(m['separation'][0]['r']/.5)} em s = 0,5 para {num(m['separation'][0]['r'])} em s = 1. Não existe um limiar universal r = 1: a forma e as direções das nuvens também importam.</p>
<p>Para sustentar a conclusão, calculei os fechos convexos de cada classe e verifiquei suas interseções pelo teorema dos eixos separadores. Em s = 0,5 todos os pares admitem uma reta separadora; em s = 1 os fechos dos pares 0–1 e 1–2 já se interceptam. Uma taxa de mistura positiva, sozinha, não provaria esse resultado. A localização exata da transição entre 0,5 e 1 não foi investigada.</p>
<p>Essa conclusão se refere às amostras geradas. Gaussianas não degeneradas têm densidade positiva em todo o plano: na população, a separação perfeita já é impossível em qualquer s &gt; 0, mesmo quando uma amostra finita parece perfeitamente separada.</p>
<h3>C. Sobreposição e fronteiras de decisão</h3>
<p>No dataset original, as classes 0 e 1 se misturam, e há também contato entre 1 e 2; os fechos convexos desses dois pares se sobrepõem. A classe 3 fica mais isolada, enquanto a classe 2 tem formato mais compacto. Uma única reta produz dois semiplanos e não define quatro regiões distintas. Um classificador multiclasses com escores lineares pode criar várias fronteiras, mas não separar perfeitamente estes pares com fechos sobrepostos.</p>
<p>O esboço da Figura 1 compara as log-densidades gaussianas conhecidas, com probabilidades de classe iguais: gₖ(x) = −½ Σⱼ ((xⱼ − μₖⱼ)/σₖⱼ)² − Σⱼ log σₖⱼ. Desenhei gᵢ = gⱼ apenas onde essas duas classes dominam as demais. As variâncias diferentes produzem curvas que uma rede com ativações não lineares poderia aproximar; nenhuma rede foi ajustada.</p>
<p>Com o aumento do espalhamento, cresce a ambiguidade probabilística entre classes. Fronteiras mais flexíveis podem reduzir erros decorrentes de uma escolha linear inadequada, mas não eliminam o erro de Bayes causado por densidades sobrepostas. “Região onde a rede necessariamente erra” significa risco irredutível de errar em novas observações, não que cada ponto dessa região precise receber o rótulo errado. Uma rede também poderia memorizar uma amostra finita. Uma combinação arbitrariamente grande de retas com decisões não lineares pode fazer regiões complexas; isso é diferente de um classificador com escores lineares.</p>
{codeblock(code[:cut1],'Código comentado — configuração e Exercício 1')}
</section>
<section id="exercicio-2">
<h2>Exercício 2 — Não linearidade em 5D</h2>
<p>Abordagem: comparar duas gaussianas com centros deslocados a uma estrutura de núcleo e casca. A PCA serve para visualizar; as medidas de distância e raio são calculadas nas cinco dimensões originais.</p>
<h3>A. Dataset I: gaussianas multivariadas</h3>
<p>Foram sorteadas 500 amostras para A e 500 para B. As médias são [0, 0, 0, 0, 0] e [1,5; 1,5; 1,5; 1,5; 1,5]. As matrizes abaixo foram usadas integralmente, e seus autovalores foram verificados como positivos.</p>
<div class="matrix-pair"><pre aria-label="Covariância da classe A">ΣA = [1,0  0,8  0,1  0,0  0,0
      0,8  1,0  0,3  0,0  0,0
      0,1  0,3  1,0  0,5  0,0
      0,0  0,0  0,5  1,0  0,2
      0,0  0,0  0,0  0,2  1,0]</pre>
<pre aria-label="Covariância da classe B">ΣB = [ 1,5 −0,7  0,2  0,0  0,0
      −0,7  1,5  0,4  0,0  0,0
       0,2  0,4  1,5  0,6  0,0
       0,0  0,0  0,6  1,5  0,3
       0,0  0,0  0,0  0,3  1,5]</pre></div>
<p>A tem correlação 0,8 entre as duas primeiras features; em B, a correlação é −0,7/1,5 ≈ −0,4667. As variâncias de B são maiores. Isso altera a orientação e o tamanho das nuvens, além do deslocamento entre centros.</p>
<h3>B. Dataset II: núcleo e casca</h3>
<p>Para cada ponto, sorteei v ∼ N(0, I₅) e normalizei u = v/‖v‖, obtendo direções uniformes na esfera unitária em ℝ⁵. Os raios são independentes das direções: ρC ∼ N(2; 0,4²) e ρD ∼ N(5; 0,4²). Cada ponto é x = ρu. Foram gerados 500 pontos por classe e todos os raios sorteados foram positivos nesta execução.</p>
<h3>C. PCA e medidas geométricas</h3>
<p>Usei PCA separada para cada dataset, com centralização e SVD completa, sem padronizar as features. As dimensões sintéticas têm a mesma unidade, e preservar suas variâncias faz parte da comparação. A PCA procura direções de maior variância e não usa os rótulos para escolher os componentes. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">Referência: PCA do scikit-learn.</a></p>
{figure(4,'Projeções dos dois datasets. Os painéis usam escalas iguais e proporção visual comparável; os eixos PCA são calculados independentemente e não representam as mesmas direções entre datasets.')}
{pca_table}
<p>Os dois primeiros componentes explicam <strong>{pct(sum(m['dataset_1']['explained_variance']))} no Dataset I</strong> e <strong>{pct(sum(m['dataset_2']['explained_variance']))} no Dataset II</strong>. A projeção preserva melhor a informação discriminativa aparente do Dataset I, cujo deslocamento entre centros contribui para a variância. No Dataset II, a informação de raio se distribui por cinco direções: eliminar três delas pode projetar pontos da casca sobre o núcleo. Variância explicada não é uma medida de acurácia.</p>
<p>As distâncias entre <em>centros amostrais</em> são <strong>{num(m['dataset_1']['center_distance'])}</strong> e <strong>{num(m['dataset_2']['center_distance'])}</strong>. Para comparação, as distâncias entre médias populacionais são √(5 × 1,5²) = {num(1.5*5**.5)} e zero, respectivamente. A pequena distância amostral do Dataset II decorre da flutuação dos sorteios.</p>
{figure(5,'Histogramas de ‖x‖ em 5D, com bins comuns às duas classes de cada dataset. As distâncias são à origem, não aos centros estimados nem aos eixos da PCA.')}
<p>No Dataset I, os raios de A variam de {num(m['dataset_1']['radius_ranges'][0][0])} a {num(m['dataset_1']['radius_ranges'][0][1])}, e os de B de {num(m['dataset_1']['radius_ranges'][1][0])} a {num(m['dataset_1']['radius_ranges'][1][1])}: os intervalos se sobrepõem. No Dataset II, C varia de {num(m['dataset_2']['radius_ranges'][0][0])} a {num(m['dataset_2']['radius_ranges'][0][1])}, e D de {num(m['dataset_2']['radius_ranges'][1][0])} a {num(m['dataset_2']['radius_ranges'][1][1])}. Há um intervalo livre entre os raios das duas classes nesta amostra.</p>
<h3>D. O que uma fronteira linear não resolve</h3>
<p>Centros praticamente coincidentes e raios separados indicam que a classe depende da distância à origem, e não de uma direção privilegiada. Um hiperplano wᵀx + b = 0 corta o espaço em dois lados. Como a casca ocupa todas as direções ao redor do núcleo, deslocar ou girar esse plano não coloca todo o núcleo de um lado e toda a casca do outro. No caso ideal de cascas simétricas, se os pontos externos x e −x ficam no mesmo semiespaço convexo, seu ponto médio, a origem, também fica nele. Coletar mais dados não muda essa estrutura.</p>
<p>Uma projeção 2D misturada <strong>não prova inseparabilidade em 5D</strong>, pois pode descartar direções ou combinações relevantes. Neste próprio Dataset II, a projeção perde a distinção clara que existe nos raios originais. Uma função não linear simples é q(x) = Σᵢ₌₁⁵ xᵢ². A regra “C se q(x) &lt; 14,44; D caso contrário”, equivalente a raio 3,8, separa os 1.000 pontos desta realização, com <strong>zero erros</strong>.</p>
<p>O limiar 3,8 foi escolhido para ilustrar o intervalo observado entre o maior raio de C ({num(m['dataset_2']['radius_ranges'][0][1])}) e o menor de D ({num(m['dataset_2']['radius_ranges'][1][0])}); não representa desempenho em teste independente. O ponto médio teórico das médias dos raios, 3,5, dá {m['radial_rule_errors']} erro em 1.000 pontos. Como os raios são gaussianos, as distribuições têm caudas sobrepostas e nem uma regra radial garante erro populacional zero.</p>
<p>Uma transformação linear seguida de outra linear continua sendo linear. Redes com ativações não lineares podem aproximar a função radial; profundidade não é condição necessária para representar toda fronteira não linear, e não remove a ambiguidade probabilística dos dados.</p>
{codeblock(code[cut1:cut2],'Código comentado — Exercício 2 (continuação)')}
</section>
<section id="exercicio-3">
<h2>Exercício 3 — Dados reais e preparação para tanh</h2>
<p>Abordagem: inspecionar o train.csv do Spaceship Titanic, reservar teste estratificado e ajustar as transformações somente no treino. O arquivo test.csv da competição não foi usado.</p>
<h3>A. Objetivo, tipos e diagnóstico dos dados</h3>
<p>A coluna <code>Transported</code> indica se um passageiro foi transportado para outra dimensão no cenário fictício da competição. O objetivo preditivo seria estimar esse evento a partir dos registros dos passageiros. Aqui apenas preparo as entradas, sem treinar classificadores. <a href="https://www.kaggle.com/competitions/spaceship-titanic/data">Fonte: descrição do dataset no Kaggle.</a></p>
<p>O arquivo contém <strong>8.693 linhas e 14 colunas</strong>. São <strong>4.378 positivos ({pct(r['positive_fraction'])})</strong> e <strong>4.315 negativos ({pct(1-r['positive_fraction'])})</strong>: as classes estão praticamente balanceadas.</p>
{table(['Tipo','Colunas'],[['Numéricas','Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck'],['Categóricas usadas','HomePlanet, CryoSleep, Destination, VIP'],['Identificadores / textos descartados','PassengerId, Cabin, Name'],['Alvo binário (não é feature)','Transported']])}
<p>CryoSleep e VIP são indicadores booleanos tratados como categorias; Cabin é uma categoria composta, Name é texto e PassengerId é um identificador. As três últimas colunas de entrada serão descartadas conforme solicitado.</p>
<p>Valores faltantes no arquivo completo; o denominador de cada percentual é 8.693:</p>
{missingtable}
<p>Estatísticas dos gastos no arquivo completo, ignorando ausentes apenas para calcular estes resumos:</p>
{stats_table(globalstats)}
<p>As medianas dos cinco gastos são zero, enquanto as médias são positivas e os máximos são muito maiores. Isso indica concentração em zero e caudas longas à direita: poucos passageiros gastam muito e deslocam a média. Um valor extremo não é, por si só, um erro de registro; por isso, os gastos altos não foram excluídos.</p>
<h3>B. Separação antes das transformações</h3>
<p>O split foi implementado com permutações estratificadas em NumPy usando o mesmo rng da atividade. Em cada classe, reservei round(20% × n) observações para teste e depois embaralhei cada partição. O resultado foi <strong>6.954 linhas de treino e 1.739 de teste</strong>, com índices disjuntos e todas as linhas originais preservadas.</p>
{table(['Partição','False','True','Total','Proporção positiva'],[['Treino',3452,3502,6954,pct(3502/6954)],['Teste',863,876,1739,pct(876/1739)]])}
<p>A separação ocorre antes de calcular qualquer estatística usada nas transformações, para que o teste represente observações não vistas. Imputar ou escalar usando o arquivo inteiro transferiria informações da distribuição do teste para o treino. Os resumos globais do item A são apenas descritivos e não alimentam imputação, categorias ou limites de escala.</p>
<p>Estatísticas de gastos <strong>somente no treino, antes de imputar e transformar</strong>:</p>
{stats_table(trainstats)}
<p>Em particular, FoodCourt no treino tem <strong>média {num(r['foodcourt_train']['mean'],4)}, mediana {num(r['foodcourt_train']['median'],2)} e máximo {num(r['foodcourt_train']['max'],2)}</strong>. A diferença entre média e mediana confirma a assimetria também na partição usada para ajustar o pré-processamento.</p>
<h3>C. Imputação, engenharia e escalonamento</h3>
<p><strong>Numéricas.</strong> Preenchi os ausentes com a mediana de cada coluna calculada no treino: Age = 27 e as cinco colunas de gasto = 0. A mediana é menos sensível à cauda longa que a média. Aplicar as mesmas medianas no teste evita vazamento. Imputar gasto zero é uma aproximação que não distingue “não gastou” de “gasto desconhecido”; foi escolhida pela concentração em zero e deve ser reavaliada em um projeto de modelagem.</p>
<p><strong>Categóricas.</strong> Os ausentes recebem o rótulo fixo “Ausente”, sem estimar frequências no teste. HomePlanet, CryoSleep, Destination e VIP passam por one-hot, com categorias aprendidas exclusivamente no treino. Uma categoria inédita no teste é representada por zeros no bloco correspondente, devido a <code>handle_unknown='ignore'</code>; ela não é confundida com o rótulo explícito “Ausente”. Essa situação foi verificada com uma categoria fictícia em uma cópia de uma linha, sem alterar as matrizes finais. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html">Referência: OneHotEncoder.</a></p>
<p><strong>TotalSpend.</strong> Após imputar os gastos, somei RoomService + FoodCourt + ShoppingMall + Spa + VRDeck na escala monetária original. Em seguida, apliquei log(1 + x) aos cinco gastos e a TotalSpend. Somar depois do log produziria outra grandeza. Cabin, Name e PassengerId não entram nas features, e Transported fica separado como alvo inteiro 0/1.</p>
<p><strong>Caudas.</strong> log(1 + x) preserva zero e a ordem dos valores, mas comprime a distância entre gastos muito elevados. Assim, valores extremos dominam menos a escala e a soma ponderada que chega à tanh. Isso reduz uma possível fonte de saturação e gradientes pequenos; não garante ausência de saturação, pois pesos e vieses também influenciam a ativação.</p>
<p><strong>Escala.</strong> Escolhi MinMax em [−1, 1], ajustado somente nas sete colunas numéricas do treino. A fórmula é z = 2(x − mínₜᵣₑᵢₙₒ)/(máxₜᵣₑᵢₙₒ − mínₜᵣₑᵢₙₒ) − 1. Os indicadores one-hot permanecem em 0/1, que também está dentro do intervalo. Uma entrada em [−1, 1] é uma escolha conveniente, não uma exigência matemática da função tanh. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">Referência: MinMaxScaler.</a></p>
<p>Usei <code>clip=True</code> para manter valores de teste além dos extremos do treino dentro do intervalo. Houve <strong>{r['clip_cells']} valor em {r['clip_rows']} passageiro</strong> limitado, na coluna RoomService. Isso não recalcula os limites com o teste, mas perde a informação de quanto o extremo excedeu o máximo de treino.</p>
{ranges}
<h3>D. Visualização, checagens e reflexão</h3>
{figure(6,'FoodCourt somente no treino: antes, após imputação e log(1 + x), e após escalonamento. Os ausentes não entram no primeiro histograma; os demais incluem todas as linhas após imputação. A concentração em zero permanece, enquanto a cauda é comprimida.')}
<p>As checagens finais encontraram <strong>0 NaN no treino e 0 NaN no teste</strong>, e nenhum valor infinito. As matrizes de features têm shapes <strong>(6.954, 21)</strong> e <strong>(1.739, 21)</strong>, respectivamente: sete numéricas e 14 indicadores categóricos. O mínimo e o máximo são <strong>−1,0000 e 1,0000 no treino</strong> e <strong>−1,0000 e 1,0000 no teste</strong>. As matrizes estão, portanto, em uma escala compatível com a proposta de uso da tanh.</p>
<p>As escolhas com maior potencial de afetar o treinamento são a transformação dos gastos e a escala das entradas, pois modificam a influência dos extremos e a faixa das pré-ativações. A imputação também importa: substituir ausências por zero pode ocultar diferenças entre desconhecimento e ausência real de consumo. One-hot evita inventar uma ordem entre planetas ou destinos, enquanto TotalSpend explicita intensidade total de consumo, embora seja redundante com os gastos originais. O split anterior ao ajuste torna uma futura avaliação mais confiável; nenhuma dessas escolhas foi validada por desempenho preditivo nesta atividade.</p>
{codeblock(code[cut2:],'Código comentado — Exercício 3 e exportação dos resultados (continuação)')}
</section>
<section id="reproducao">
<h2>Reprodução e origem dos arquivos</h2>
<p>Execute os comandos abaixo na raiz do repositório com Python 3.13. O primeiro script reinicia uma única sequência com semente 42 e gera as análises na ordem do relatório. O segundo monta esta página usando os resultados calculados.</p>
<pre><code>python -m pip install -r requirements.txt
python scripts/analyze.py
python scripts/build_report.py
python scripts/build_site.py</code></pre>
<p>Bibliotecas científicas: NumPy 2.4.3, pandas 2.3.3, Matplotlib 3.10.9 e scikit-learn 1.8.0. Do scikit-learn foram usados apenas PCA e pré-processamento. A biblioteca padrão de Python é usada para arquivos, HTML e verificação de integridade.</p>
<p>O download oficial do Kaggle respondeu HTTP 401 neste ambiente. Foi utilizada uma cópia pública de <a href="https://github.com/You-sha/Spaceship-Titanic/blob/main/train.csv">train.csv</a>, conferida byte a byte com <a href="https://github.com/AmirFARES/Kaggle-Spaceship-Titanic/blob/main/data/train.csv">um segundo espelho</a>. Ambos os arquivos são idênticos. A origem conceitual continua sendo a competição <a href="https://www.kaggle.com/competitions/spaceship-titanic/data">Spaceship Titanic</a>; não foi possível comparar os bytes com um download autenticado do Kaggle.</p>
<p>SHA-256 do arquivo efetivamente analisado:</p><pre><code>{r['sha256']}</code></pre>
<p><a href="results/features.npz" download>Baixar matrizes, alvos, índices e nomes de features</a> · <a href="results/faltantes.csv">Tabela de faltantes</a> · <a href="results/gastos-treino.csv">Estatísticas de gastos no treino</a> · <a href="https://github.com/henriquessm/ann-dl">Repositório público</a>.</p>
<p>Colaboração com IA: apoio na implementação, visualização e redação. Resultados obtidos por execução do código disponibilizado, sem treinamento de modelos.</p>
</section>
<section id="resumo">
<h2>Resumo dos resultados</h2>
<p>Índice dos 13 resultados solicitados. As distâncias de centros dos datasets I e II usam as médias amostrais em cinco dimensões.</p>
{table(['#','Item','Seu valor'],[[i,*row] for i,row in enumerate(summary_rows,1)])}
</section>
'''
front='---\nlayout: default\ntitle: "Preparação e análise de dados para redes neurais"\nlang: pt-BR\n---\n'
(ROOT/'index.html').write_text(front+body,encoding='utf-8')
print('Relatório gerado: index.html')
