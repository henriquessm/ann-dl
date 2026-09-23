"""Exercício Perceptron: dados, modelo, treino (com pocket) e Figuras 1 a 6.

Executar a partir da raiz do repositório:
    python docs/exercises/perceptron/code/perceptron.py
Escreve as figuras em ../figures e todos os números reportados em results.json.
Um único gerador (rng) é usado do início ao fim; execute o script inteiro para reproduzir.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
ETA, MAX_EPOCHS = 0.01, 100
COLORS = {0: "#2a78d6", 1: "#eb6834"}  # classe 0 azul, classe 1 laranja


def make_data(mean0, mean1, cov, n=1000):
    """Duas classes gaussianas 2D com n amostras cada; rótulos 0 e 1."""
    x0 = rng.multivariate_normal(mean0, cov, size=n)
    x1 = rng.multivariate_normal(mean1, cov, size=n)
    X = np.vstack([x0, x1])
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    return X, y


def step(z):
    """Ativação degrau: 1 se z >= 0, 0 caso contrário."""
    return (z >= 0).astype(int)


def predict(X, w, b):
    """ŷ = step(w·x + b) para todas as linhas de X."""
    return step(X @ w + b)


def accuracy(X, y, w, b):
    return float(np.mean(predict(X, w, b) == y))


def train(X, y, eta=ETA, max_epochs=MAX_EPOCHS, w0=None, b0=0.0):
    """Perceptron de camada única com a regra w += eta*(y-ŷ)x, b += eta*(y-ŷ).

    Para em uma época sem nenhuma atualização ou em max_epochs.
    Devolve os pesos finais, a curva de acurácia por época e o "pocket":
    os melhores pesos vistos após qualquer atualização (algoritmo pocket).
    Em dados separáveis o pocket coincide com os pesos finais.
    """
    w = rng.normal(0, 0.01, size=2) if w0 is None else np.array(w0, float)
    b = float(b0)
    pocket = {"w": w.copy(), "b": b, "acc": accuracy(X, y, w, b), "epoch": 0}
    acc_epoch, best_epoch, updates_epoch = [], [], []
    epochs = 0
    for epoch in range(1, max_epochs + 1):
        n_updates = 0
        for xi, yi in zip(X, y):
            err = yi - step(xi @ w + b)
            if err != 0:
                w = w + eta * err * xi
                b = b + eta * err
                n_updates += 1
                acc = accuracy(X, y, w, b)
                if acc > pocket["acc"]:
                    pocket = {"w": w.copy(), "b": b, "acc": acc, "epoch": epoch}
        epochs = epoch
        acc_epoch.append(accuracy(X, y, w, b))
        best_epoch.append(pocket["acc"])
        updates_epoch.append(n_updates)
        if n_updates == 0:
            break
    return {
        "w": w, "b": b, "epochs": epochs, "acc": acc_epoch[-1],
        "acc_epoch": acc_epoch, "best_epoch": best_epoch,
        "updates_epoch": updates_epoch, "pocket": pocket,
    }


def scatter(ax, X, y):
    for c in (0, 1):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=9, c=COLORS[c], alpha=0.65,
                   edgecolors="none", label=f"Classe {c}")


def boundary(ax, w, b, X, **kw):
    """Reta w·x + b = 0 no intervalo de x1 dos dados."""
    x1 = np.array([X[:, 0].min(), X[:, 0].max()])
    if abs(w[1]) > 1e-12:
        ax.plot(x1, -(w[0] * x1 + b) / w[1], **kw)
    else:
        ax.axvline(-b / w[0], **kw)


def mark_errors(ax, X, y, w, b):
    """Anel preto vazado sobre cada ponto classificado errado por (w, b)."""
    wrong = predict(X, w, b) != y
    ax.scatter(X[wrong, 0], X[wrong, 1], s=30, facecolors="none", edgecolors="black",
               linewidths=0.7, label=f"Erros ({wrong.sum()} de {len(y)})")


def fig_scatter(n, X, y, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter(ax, X, y)
    ax.set(title=title, xlabel="x₁", ylabel="x₂")
    ax.legend()
    fig.savefig(FIG / f"fig{n}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_boundary(n, X, y, panels, title):
    """Um painel por fronteira; panels: lista de (w, b, rótulo da reta, estilo)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6), sharey=True, squeeze=False)
    pad = 0.5
    for ax, (w, b, lab, style) in zip(axes[0], panels):
        scatter(ax, X, y)
        boundary(ax, w, b, X, label=lab, lw=2, **style)
        mark_errors(ax, X, y, w, b)
        ax.set(title=lab, xlabel="x₁", ylabel="x₂",
               xlim=(X[:, 0].min() - pad, X[:, 0].max() + pad),
               ylim=(X[:, 1].min() - pad, X[:, 1].max() + pad))
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.savefig(FIG / f"fig{n}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_curve(n, curves, title):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for values, lab, style in curves:
        ax.plot(range(1, len(values) + 1), np.array(values) * 100, lw=2,
                label=lab, **style)
    ax.set(title=title, xlabel="Época", ylabel="Acurácia no dataset completo (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(FIG / f"fig{n}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary(r):
    """Números reportados no texto para uma execução de train()."""
    return {"w": r["w"].tolist(), "b": r["b"], "epochs": r["epochs"], "acc": r["acc"],
            "norm_w": float(np.linalg.norm(r["w"])),
            "unit_w": (r["w"] / np.linalg.norm(r["w"])).tolist(),
            "updates_epoch": r["updates_epoch"], "acc_epoch": r["acc_epoch"]}


if __name__ == "__main__":
    out = {}

    X1, y1 = make_data([1.5, 1.5], [5, 5], [[0.5, 0], [0, 0.5]])
    fig_scatter(1, X1, y1, "Figura 1 — Dados separáveis (1000 pontos por classe)")
    w0 = rng.normal(0, 0.01, size=2)
    r1 = train(X1, y1, eta=0.01, w0=w0)
    r1b = train(X1, y1, eta=1.0, w0=w0)
    fig_boundary(2, X1, y1,
                 [(r1["w"], r1["b"], "Fronteira w·x + b = 0 (η = 0,01)", {"c": "black"})],
                 "Figura 2 — Fronteira de decisão, dados separáveis")
    fig_curve(3, [(r1["acc_epoch"], "Acurácia por época (η = 0,01)", {"c": COLORS[0], "marker": "o"})],
              "Figura 3 — Acurácia × época, dados separáveis")
    z1 = train(X1, y1, eta=0.01, w0=[0, 0])
    z2 = train(X1, y1, eta=1.0, w0=[0, 0])
    out["ex1"] = {
        "w0": w0.tolist(), "eta_0.01": summary(r1), "eta_1.0": summary(r1b),
        "zero_start": {"eta_0.01": summary(z1), "eta_1.0": summary(z2),
                       "ratio_w": (z2["w"] / z1["w"]).tolist(), "ratio_b": z2["b"] / z1["b"]},
    }

    X2, y2 = make_data([3, 3], [4, 4], [[1.5, 0], [0, 1.5]])
    fig_scatter(4, X2, y2, "Figura 4 — Dados sobrepostos (1000 pontos por classe)")
    r2 = train(X2, y2)
    p = r2["pocket"]
    fig_boundary(5, X2, y2,
                 [(r2["w"], r2["b"], "Fronteira final (época 100)", {"c": "black"}),
                  (p["w"], p["b"], f"Fronteira pocket (época {p['epoch']})", {"c": "#008300", "ls": "--"})],
                 "Figura 5 — Fronteiras final e pocket, dados sobrepostos")
    fig_curve(6, [(r2["acc_epoch"], "Pesos atuais (fim da época)", {"c": COLORS[0]}),
                  (r2["best_epoch"], "Melhor até agora (pocket)", {"c": "#008300", "ls": "--"})],
              "Figura 6 — Acurácia × época, dados sobrepostos")
    out["ex2"] = {
        "final": summary(r2),
        "pocket": {"w": p["w"].tolist(), "b": p["b"], "acc": p["acc"], "epoch": p["epoch"]},
        "best_epoch": r2["best_epoch"],
        "mean_norm_x": float(np.linalg.norm(X2, axis=1).mean()),
        "pred1_fraction_final": float(predict(X2, r2["w"], r2["b"]).mean()),
        "acc_range_epochs": [min(r2["acc_epoch"]), max(r2["acc_epoch"])],
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if not kk.endswith("epoch")}, indent=1))
