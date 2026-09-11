"""Gera todas as análises e figuras, sem treinar modelos. Execute da raiz do repo."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'assets' / 'figures'
RES = ROOT / 'results'
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(exist_ok=True)
# Única fonte de aleatoriedade: não reinicializar entre exercícios.
rng = np.random.default_rng(42)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff',
                     'axes.titleweight': 'bold', 'savefig.dpi': 160})
COLORS = ['#176b9a', '#dc8036', '#7b54a5', '#28836a']
metrics = {}

def save(fig, number):
    fig.savefig(FIG / f'figura-{number}.png', bbox_inches='tight')
    plt.close(fig)

def scatter(ax, x, y, names, centers=None):
    for k, name in enumerate(names):
        pts = x[y == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=15, color=COLORS[k], alpha=.65, label=name)
        if centers is not None:
            ax.scatter(*centers[k], marker='X', s=140, color=COLORS[k], edgecolor='black', zorder=5)
    ax.legend(fontsize=8)
    ax.set(xlabel='Feature x', ylabel='Feature y')
    ax.grid(alpha=.12)

# Exercício 1 A: gaussianas independentes em x e y.
means = np.array([[2, 3], [5, 6], [8, 1], [15, 4]], dtype=float)
stds = np.array([[.8, 2.5], [1.2, 1.9], [.9, .9], [.5, 2.]])
labels = np.repeat(np.arange(4), 100)
def clouds(scale):
    return np.vstack([rng.normal(m, scale * sd, size=(100, 2)) for m, sd in zip(means, stds)])

# Teste exato até tolerância numérica: fechos convexos e eixos separadores.
# Não ajusta classificador. Fechos convexos que se interceptam impedem separação estrita por reta.
def hull(points):
    p = sorted(set(map(tuple, points)))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])

def intersect(a, b):
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        axes = np.column_stack([-edges[:, 1], edges[:, 0]])
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        pa, pb = a @ axes.T, b @ axes.T
        if np.any(pa.max(axis=0) < pb.min(axis=0)-1e-10) or np.any(pb.max(axis=0) < pa.min(axis=0)-1e-10):
            return False
    return True

def overlaps(x):
    hs = [hull(x[labels == k]) for k in range(4)]
    return [f'{i}–{j}' for i in range(4) for j in range(i+1, 4) if intersect(hs[i], hs[j])]

original = clouds(1)
metrics['original_overlap_pairs'] = overlaps(original)
fig, ax = plt.subplots(figsize=(10, 6))
scatter(ax, original, labels, [f'Classe {k}' for k in range(4)], means)
# Esboço informado pelas densidades geradoras conhecidas, com priors iguais.
# Não é uma rede ajustada: fronteiras quadráticas ilustram decisões plausíveis.
lo, hi = original.min(axis=0)-1, original.max(axis=0)+1
gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 450), np.linspace(lo[1], hi[1], 450))
grid = np.column_stack([gx.ravel(), gy.ravel()])
scores = -.5 * (((grid[:, None, :] - means) / stds)**2).sum(axis=2) - np.log(stds).sum(axis=1)
for i in range(4):
    for j in range(i+1, 4):
        # Desenhar igualdade apenas onde i e j superam as outras classes.
        others = [k for k in range(4) if k not in (i, j)]
        delta = scores[:, i]-scores[:, j]
        relevant = np.maximum(scores[:, i], scores[:, j]) >= scores[:, others].max(axis=1)
        z = np.ma.masked_where(~relevant.reshape(gx.shape), delta.reshape(gx.shape))
        if z.count() and z.min() < 0 < z.max():
            ax.contour(gx, gy, z, levels=[0], colors='#344354', linewidths=1.2, linestyles='--')
handles, names = ax.get_legend_handles_labels()
handles += [Line2D([], [], color='black', marker='X', linestyle='None', label='Médias geradoras'),
            Line2D([], [], color='#344354', linestyle='--', label='Esboço analítico (sem treino)')]
ax.legend(handles=handles, fontsize=8, loc='upper right')
ax.set(title='Figura 1 · Quatro nuvens e fronteiras plausíveis', xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
save(fig, 1)

# Exercício 1 B: quatro novas realizações independentes; mesmas médias.
scales = [.5, 1., 2., 4.]
datasets = [clouds(s) for s in scales]
rates = [float((np.linalg.norm(x[:, None, :] - means, axis=2).argmin(axis=1) != labels).mean()) for x in datasets]
metrics['mixture'] = dict(zip(map(str, scales), rates))
metrics['overlap_pairs'] = {str(s): overlaps(x) for s, x in zip(scales, datasets)}
pairs = [{'par': f'{i}–{j}', 'r': float(np.linalg.norm(means[i]-means[j]) / (stds[i].mean()+stds[j].mean()))}
         for i in range(4) for j in range(i+1, 4)]
metrics['separation'] = pairs
fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
all_points = np.vstack(datasets)
lo, hi = all_points.min(axis=0)-1, all_points.max(axis=0)+1
for ax, s, x, rate in zip(axs.flat, scales, datasets, rates):
    scatter(ax, x, labels, [f'Classe {k}' for k in range(4)], means)
    ax.set(title=f's = {s:.1f} | mistura = {rate:.2%}', xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
fig.suptitle('Figura 2 · O efeito do espalhamento, na mesma escala', fontsize=15)
fig.tight_layout()
save(fig, 2)
fig, ax = plt.subplots(figsize=(9, 4.7))
ax.plot(scales, np.array(rates)*100, 'o-', color=COLORS[0], label='Classes 0–3 (taxa agregada)')
for s, rate in zip(scales, rates):
    ax.annotate(f'{rate:.2%}', (s, rate*100), xytext=(0, 9), textcoords='offset points', ha='center')
ax.set(title='Figura 3 · Taxa de mistura × escala', xlabel='Fator de escala s', ylabel='Pontos com outro centro mais próximo (%)', xticks=scales, ylim=(0, max(rates)*100+7))
ax.legend(); ax.grid(alpha=.15)
save(fig, 3)

# Exercício 2 A e B: gerar em 5D, preservar as distâncias no espaço original.
cov_a = np.array([[1,.8,.1,0,0],[.8,1,.3,0,0],[.1,.3,1,.5,0],[0,0,.5,1,.2],[0,0,0,.2,1]])
cov_b = np.array([[1.5,-.7,.2,0,0],[-.7,1.5,.4,0,0],[.2,.4,1.5,.6,0],[0,0,.6,1.5,.3],[0,0,0,.3,1.5]])
assert np.linalg.eigvalsh(cov_a).min() > 0 and np.linalg.eigvalsh(cov_b).min() > 0
xa = rng.multivariate_normal(np.zeros(5), cov_a, 500)
xb = rng.multivariate_normal(np.full(5, 1.5), cov_b, 500)
def radial(mu):
    directions = rng.normal(size=(500, 5))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.normal(mu, .4, size=500)
    assert (radii > 0).all()  # Nesta realização, raio gaussiano positivo em todos os pontos.
    return directions * radii[:, None]
xc, xd = radial(2), radial(5)
yl = np.repeat([0, 1], 500)
fig4, axs4 = plt.subplots(1, 2, figsize=(12, 5))
fig5, axs5 = plt.subplots(1, 2, figsize=(12, 4.7))
for idx, (a, b, names) in enumerate([(xa, xb, ['Classe A', 'Classe B']), (xc, xd, ['Classe C', 'Classe D'])]):
    x = np.vstack([a, b])
    pca = PCA(n_components=2, svd_solver='full')  # SVD determinística; sem outro RNG.
    proj = pca.fit_transform(x)
    ev = pca.explained_variance_ratio_
    radii = [np.linalg.norm(z, axis=1) for z in (a, b)]
    metrics[f'dataset_{idx+1}'] = {'center_distance': float(np.linalg.norm(a.mean(axis=0)-b.mean(axis=0))),
        'explained_variance': ev.tolist(), 'radius_ranges': [[float(r.min()),float(r.max())] for r in radii]}
    scatter(axs4[idx], proj, yl, names)
    axs4[idx].set(title=f'Dataset {"I" if idx == 0 else "II"} | PC1 + PC2 = {ev.sum():.2%}', xlabel=f'PC1 ({ev[0]:.2%})', ylabel=f'PC2 ({ev[1]:.2%})', xlim=(-7,7), ylim=(-7,7))
    bins = np.linspace(0, max(r.max() for r in radii)+.1, 35)
    for k, r in enumerate(radii):
        axs5[idx].hist(r, bins=bins, alpha=.6, color=COLORS[k], label=names[k], edgecolor='white', linewidth=.3)
    axs5[idx].set(title=f'Dataset {"I" if idx == 0 else "II"} · raios em 5D', xlabel='Raio ‖x‖ no espaço original', ylabel='Número de pontos')
    axs5[idx].legend()
fig4.suptitle('Figura 4 · Projeções PCA em duas dimensões', fontsize=15)
fig4.tight_layout(); save(fig4, 4)
fig5.suptitle('Figura 5 · Distâncias à origem, por classe', fontsize=15)
fig5.tight_layout(); save(fig5, 5)
metrics['radial_rule_errors'] = int((np.linalg.norm(xc,axis=1) >= 3.5).sum() + (np.linalg.norm(xd,axis=1) < 3.5).sum())
metrics['radial_rule_3_8_errors'] = int((np.linalg.norm(xc,axis=1) >= 3.8).sum() + (np.linalg.norm(xd,axis=1) < 3.8).sum())

# Exercício 3: único arquivo com rótulos; verificar integridade e separar ANTES das estatísticas.
data_path = ROOT / 'data/train.csv'
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == '17336d553f49ebdf6ecb266d2b5d3746e5dd308445f7c7864141c4f28d2a88d0', 'O arquivo de entrada difere da cópia verificada.'
df = pd.read_csv(data_path)
assert df.shape == (8693, 14) and df.PassengerId.is_unique
assert not df.Transported.isna().any()
y = df.Transported.astype(int)
spend = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
numeric = ['Age'] + spend
categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
# Split estratificado com o MESMO rng: permutar cada classe e reservar ~20%.
train_idx, test_idx = [], []
for c in (0, 1):
    ids = rng.permutation(np.flatnonzero(y.to_numpy() == c))
    ntest = round(.2 * len(ids))
    test_idx.extend(ids[:ntest]); train_idx.extend(ids[ntest:])
train_idx = rng.permutation(train_idx); test_idx = rng.permutation(test_idx)
train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
assert len(set(train_idx) & set(test_idx)) == 0
assert len(train) + len(test) == len(df)

# Estatísticas descritivas globais são reportadas, mas nunca entram no pipeline.
missing = pd.DataFrame({'Coluna': df.columns, 'Faltantes': df.isna().sum().values, 'Percentual': df.isna().mean().values*100})
missing.to_csv(RES / 'faltantes.csv', index=False)
stats_all = df[spend].agg(['mean','median','max']).T
stats_train = train[spend].agg(['mean','median','max']).T
stats_all.to_csv(RES / 'gastos-global.csv')
stats_train.to_csv(RES / 'gastos-treino.csv')

# Numéricas: medianas calculadas SÓ no treino, robustas à cauda longa.
medians = train[numeric].median()
num_train, num_test = train[numeric].fillna(medians), test[numeric].fillna(medians)
# TotalSpend é soma monetária DEPOIS da imputação, ANTES de log1p.
for frame in (num_train, num_test):
    frame['TotalSpend'] = frame[spend].sum(axis=1)
    frame[spend + ['TotalSpend']] = np.log1p(frame[spend + ['TotalSpend']])

# Categóricas: rótulo explícito para ausentes; a ausência pode ser informativa.
# Categorias inéditas recebem vetor zero no bloco correspondente.
cat_train = train[categorical].astype('string').fillna('Ausente')
cat_test = test[categorical].astype('string').fillna('Ausente')
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64)
encoded_train = encoder.fit_transform(cat_train)
encoded_test = encoder.transform(cat_test)

# MinMax ajustado só no treino. Clip controla valores de teste fora dos extremos observados.
scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
scaled_train = scaler.fit_transform(num_train)
outside = (num_test.to_numpy() < scaler.data_min_) | (num_test.to_numpy() > scaler.data_max_)
scaled_test = scaler.transform(num_test)
feature_names = list(num_train.columns) + encoder.get_feature_names_out(categorical).tolist()
X_train = np.column_stack([scaled_train, encoded_train])
X_test = np.column_stack([scaled_test, encoded_test])
assert np.isfinite(X_train).all() and np.isfinite(X_test).all()
assert X_train.min() >= -1-1e-12 and X_train.max() <= 1+1e-12
assert X_test.min() >= -1-1e-12 and X_test.max() <= 1+1e-12
unknown_probe = cat_test.iloc[:1].copy()
unknown_probe.loc[:, 'HomePlanet'] = 'Planeta_nunca_visto'
assert encoder.transform(unknown_probe)[0, :len(encoder.categories_[0])].sum() == 0
np.savez_compressed(RES / 'features.npz', X_train=X_train, X_test=X_test,
    y_train=y.iloc[train_idx].to_numpy(), y_test=y.iloc[test_idx].to_numpy(),
    train_index=train_idx, test_index=test_idx, feature_names=np.array(feature_names))
metrics['real'] = {'sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
    'n': len(df), 'positive': int(y.sum()), 'negative': int((1-y).sum()), 'positive_fraction': float(y.mean()),
    'train_positive': int(train.Transported.sum()), 'test_positive': int(test.Transported.sum()),
    'foodcourt_train': stats_train.loc['FoodCourt'].to_dict(),
    'shape_train': list(X_train.shape), 'shape_test': list(X_test.shape),
    'range_train': [float(X_train.min()),float(X_train.max())], 'range_test':[float(X_test.min()),float(X_test.max())],
    'nan_train': int(np.isnan(X_train).sum()), 'nan_test':int(np.isnan(X_test).sum()),
    'clip_cells': int(outside.sum()), 'clip_rows':int(outside.any(axis=1).sum()),
    'clip_by_feature':dict(zip(num_train.columns, outside.sum(axis=0).tolist())),
    'medians':medians.to_dict(), 'features':feature_names,
    'categories': {c:v.tolist() for c,v in zip(categorical,encoder.categories_)},
    'numeric_ranges': {c:{'train_min':float(scaled_train[:,i].min()),'train_max':float(scaled_train[:,i].max()),
        'test_min':float(scaled_test[:,i].min()),'test_max':float(scaled_test[:,i].max())} for i,c in enumerate(num_train.columns)}}

# Figura 6: três etapas, cada uma com as classes do treino sobrepostas.
fig, axs = plt.subplots(1, 3, figsize=(14, 4.7))
ys = train.Transported.to_numpy(dtype=int)
values = [train.FoodCourt.to_numpy(), num_train.FoodCourt.to_numpy(), scaled_train[:, list(num_train.columns).index('FoodCourt')]]
titles = ['Antes: valores observados', 'Após imputação + log(1 + x)', 'Após MinMax em [−1, 1]']
xlabels = ['Gasto FoodCourt (unidade do dataset)', 'log(1 + FoodCourt)', 'FoodCourt escalonado']
for ax, vals, title, xlabel in zip(axs, values, titles, xlabels):
    bins = np.linspace(np.nanmin(vals), np.nanmax(vals), 36)
    for k in (0, 1):
        subset = vals[ys == k]; subset = subset[np.isfinite(subset)]
        ax.hist(subset, bins=bins, alpha=.6, color=COLORS[k], label=f'Transported = {bool(k)}')
    ax.set(title=title, xlabel=xlabel, ylabel='Número de passageiros (treino)')
    ax.legend(fontsize=8)
axs[-1].set_xticks([-1, -.5, 0, .5, 1])
fig.suptitle('Figura 6 · FoodCourt: cauda pesada e transformação', fontsize=15)
fig.tight_layout(); save(fig, 6)
(RES / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(metrics, indent=2, ensure_ascii=False))
