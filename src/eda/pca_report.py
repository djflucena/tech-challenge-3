# src/eda/pca_report.py
"""
Relatórios de PCA para interpretação e apoio à seleção de variáveis.

Este módulo **não** salva artefatos: as funções retornam objetos (DataFrames/arrays)
e/ou produzem gráficos diretamente com matplotlib, para uso no notebook.

Exemplo rápido de uso no notebook:
----------------------------------
>>> import pandas as pd
>>> from src.eda.pca_report import (
...     prepare_xy, pca_summary, plot_scree, plot_scores,
...     plot_correlation_circle, plot_biplot,
...     rank_features, prune_by_correlation
... )
>>> df = pd.read_csv("data/raw/wdbc.csv")
>>> X_df, y = prepare_xy(df, target="diagnosis", positive_label="Maligno")
>>> pca_out = pca_summary(X_df, y)
>>> plot_scree(pca_out["expl_var"])
>>> plot_scores(pca_out["scores"], y, pca_out["expl_var"])
>>> plot_correlation_circle(pca_out["loadings"], pca_out["feature_names"], pca_out["expl_var"])
>>> plot_biplot(pca_out["scores"], pca_out["loadings"], pca_out["feature_names"], pca_out["expl_var"])
>>> rank_df = rank_features(X_df, y, pca_out["contrib_df"])
>>> selected = prune_by_correlation(X_df, priority=rank_df.index.tolist(), threshold=0.90, method="spearman")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif


__all__ = [
    "prepare_xy",
    "pca_summary",
    "plot_scree",
    "plot_scores",
    "plot_correlation_circle",
    "plot_biplot",
    "rank_features",
    "prune_by_correlation",
]


# ------------------------------- Utils ------------------------------------- #

def _ellipse_line(x: np.ndarray, y: np.ndarray, n_std: float = 2.0, num: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula os pontos (x,y) de uma elipse de confiança (~n_std desvios) para um conjunto 2D.

    Args:
        x: coordenadas no eixo 1 (array 1D).
        y: coordenadas no eixo 2 (array 1D).
        n_std: múltiplo do desvio-padrão para a elipse.
        num: número de pontos para desenhar a elipse.

    Returns:
        (ex, ey): arrays 1D de tamanho `num` com a elipse rotacionada e centrada na média.
    """
    x = np.asarray(x); y = np.asarray(y)
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.arctan2(*vecs[:, 0][::-1])
    t = np.linspace(0, 2 * np.pi, num)
    w, h = n_std * np.sqrt(vals[0]), n_std * np.sqrt(vals[1])
    ellipse = np.array([w * np.cos(t), h * np.sin(t)])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    e = R @ ellipse
    return e[0] + x.mean(), e[1] + y.mean()


def _contrib(load_col: np.ndarray) -> np.ndarray:
    """
    Contribuições percentuais de uma coluna de *loadings* para um PC.
    """
    w2 = load_col ** 2
    return 100.0 * w2 / w2.sum()


# ------------------------------- API --------------------------------------- #

def prepare_xy(
    df: pd.DataFrame,
    target: str = "diagnosis",
    positive_label: str = "Maligno",
    numeric_only: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Separa X (features) e y (binário) a partir de um DataFrame do projeto.

    Args:
        df: DataFrame com a coluna-alvo e as features.
        target: nome da coluna-alvo (padrão: "diagnosis").
        positive_label: rótulo considerado classe positiva (1).
        numeric_only: se True, retorna apenas colunas numéricas em X.

    Returns:
        (X_df, y):
            - X_df: DataFrame de features (por padrão, apenas numéricas).
            - y: array 1D com 0/1 (1 = classe positiva).

    Raises:
        KeyError: se `target` não existir em df.
    """
    if target not in df.columns:
        raise KeyError(f"coluna alvo '{target}' não encontrada.")
    y = (df[target] == positive_label).astype(int).to_numpy()
    X_df = df.drop(columns=[target])
    if numeric_only:
        X_df = X_df.select_dtypes(include=[np.number])
    return X_df, y


def pca_summary(
    X_df: pd.DataFrame,
    y: Optional[np.ndarray] = None,
    standardize: bool = True,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Executa PCA para **interpretação** (não redução) e retorna resumos úteis.

    Args:
        X_df: DataFrame de features numéricas.
        y: (opcional) rótulos binários (usado apenas para contexto; não afeta PCA).
        standardize: se True, aplica StandardScaler antes do PCA.
        random_state: semente para reprodutibilidade do PCA.

    Returns:
        dict com:
            - "pca": objeto sklearn.decomposition.PCA
            - "scores": matriz de escores (n_amostras × n_features)
            - "expl_var": variância explicada por componente (%)
            - "cum": variância explicada cumulativa (%)
            - "n95": número de PCs para cobrir ≥95% da variância
            - "loadings": matriz de *loadings* (p × k)
            - "feature_names": lista de nomes das variáveis
            - "contrib_df": DataFrame com contribuições/cos² e cargas (PC1/PC2)

    Example:
        >>> import pandas as pd, numpy as np
        >>> X_df = pd.DataFrame({"a":[1,2,3], "b":[1,2,4], "c":[10,9,8]})
        >>> out = pca_summary(X_df)
        >>> "expl_var" in out and "loadings" in out
        True
    """
    feature_names = X_df.columns.to_list()
    X = X_df.to_numpy()

    if standardize:
        X_std = StandardScaler().fit_transform(X)
    else:
        X_std = X

    pca = PCA(n_components=X_std.shape[1], random_state=random_state)
    scores = pca.fit_transform(X_std)

    expl_var = pca.explained_variance_ratio_ * 100.0
    cum = np.cumsum(expl_var)
    n95 = int(np.argmax(cum >= 95.0) + 1)

    # Loadings / correlações variável–PC
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    pc1, pc2 = loadings[:, 0], loadings[:, 1]
    contrib_pc1 = _contrib(pc1)
    contrib_pc2 = _contrib(pc2)

    contrib_df = pd.DataFrame(
        {
            "contrib_PC1": contrib_pc1,
            "contrib_PC2": contrib_pc2,
            "cos2_PC1": pc1 ** 2,
            "cos2_PC2": pc2 ** 2,
            "loading_PC1": pc1,
            "loading_PC2": pc2,
        },
        index=feature_names,
    )

    return dict(
        pca=pca,
        scores=scores,
        expl_var=expl_var,
        cum=cum,
        n95=n95,
        loadings=loadings,
        feature_names=feature_names,
        contrib_df=contrib_df,
    )


def plot_scree(expl_var: np.ndarray, top_k: int = 10) -> None:
    """
    Desenha o Scree plot (variância explicada por componente).

    Args:
        expl_var: array 1D com variância explicada (%) por PC.
        top_k: número de PCs a mostrar (default: 10; usa <= se houver menos).

    Produz:
        Um gráfico de barras + linha com os primeiros `top_k` PCs.
    """
    k = min(top_k, len(expl_var))
    xs = np.arange(1, k + 1)
    fig = plt.figure()
    plt.bar(xs, expl_var[:k])
    plt.plot(xs, expl_var[:k], marker="o")
    for i, v in enumerate(expl_var[:k], 1):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.xlabel("Componentes")
    plt.ylabel("Variância explicada (%)")
    plt.title("Scree plot")
    plt.tight_layout()
    plt.show()


def plot_scores(
    scores: np.ndarray,
    y: np.ndarray,
    expl_var: np.ndarray,
    n_std: float = 2.0,
) -> None:
    """
    Dispersão PC1×PC2 por classe, com elipses (~n_std desvios).

    Args:
        scores: matriz de escores (n×k) do PCA.
        y: rótulos binários (0/1) ou categóricos com 2 classes.
        expl_var: variância explicada (%) para legendas dos eixos.
        n_std: abertura da elipse (em desvios-padrão).

    Produz:
        Um scatter plot com duas classes e elipses de dispersão.
    """
    fig, ax = plt.subplots()
    labels = np.unique(y)
    markers = ["o", "^"]
    names = ["Benigno", "Maligno"] if set(labels) == {0, 1} else [str(l) for l in labels]

    for idx, (lab, marker) in enumerate(zip(labels, markers)):
        sel = (y == lab)
        ax.scatter(scores[sel, 0], scores[sel, 1], s=25, alpha=0.6, label=names[idx], marker=marker)
        ex, ey = _ellipse_line(scores[sel, 0], scores[sel, 1], n_std=n_std)
        ax.plot(ex, ey, alpha=0.5)

    ax.axhline(0, ls="--", lw=0.8); ax.axvline(0, ls="--", lw=0.8)
    ax.set_xlabel(f"Dim1 (PC1) — {expl_var[0]:.1f}%")
    ax.set_ylabel(f"Dim2 (PC2) — {expl_var[1]:.1f}%")
    ax.set_title("Dispersão (PCA)")
    ax.legend(title="Classe")
    plt.tight_layout()
    plt.show()


def plot_correlation_circle(
    loadings: np.ndarray,
    feature_names: List[str],
    expl_var: np.ndarray,
    thickness_by: str = "cos2",
) -> None:
    """
    Círculo de correlações (PC1 vs PC2) com setas por variável.

    Args:
        loadings: matriz p×k de *loadings* do PCA.
        feature_names: nomes das variáveis (tamanho p).
        expl_var: variância explicada (%), para eixos.
        thickness_by: "cos2" ou "none" — regula a espessura das setas.

    Produz:
        Um gráfico com círculo unidade e setas das variáveis.
    """
    pc1, pc2 = loadings[:, 0], loadings[:, 1]
    cos2 = pc1 ** 2 + pc2 ** 2

    theta = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots()
    ax.plot(np.cos(theta), np.sin(theta), lw=1.0)

    for i, feat in enumerate(feature_names):
        lw = 1.0 + 4.0 * (cos2[i] / cos2.max()) if thickness_by == "cos2" else 1.25
        ax.arrow(0, 0, pc1[i], pc2[i], head_width=0.02, head_length=0.03,
                 length_includes_head=True, linewidth=lw, alpha=0.85)
        ax.text(pc1[i] * 1.08, pc2[i] * 1.08, feat, fontsize=7, ha="center", va="center")

    ax.axhline(0, ls="--", lw=0.8); ax.axvline(0, ls="--", lw=0.8)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(f"Dim1 (PC1) — {expl_var[0]:.1f}%")
    ax.set_ylabel(f"Dim2 (PC2) — {expl_var[1]:.1f}%")
    ax.set_title("PCA — círculo de correlações")
    plt.tight_layout()
    plt.show()


def plot_biplot(
    scores: np.ndarray,
    loadings: np.ndarray,
    feature_names: List[str],
    expl_var: np.ndarray,
    scale: Optional[float] = None,
) -> None:
    """
    Biplot: observações (PC1×PC2) + setas de variáveis escaladas.

    Args:
        scores: escores do PCA (n×k).
        loadings: *loadings* (p×k).
        feature_names: nomes das variáveis (p).
        expl_var: variância explicada (%) por PC.
        scale: fator opcional para escalonar setas; se None, usa 0.75*max|scores|.

    Produz:
        Um gráfico único com pontos (sem classe) e setas de variáveis.
    """
    if scale is None:
        scale = 0.75 * np.max(np.abs(scores[:, :2]))

    fig, ax = plt.subplots()
    ax.scatter(scores[:, 0], scores[:, 1], s=20, alpha=0.5, label="Observações")

    for i, feat in enumerate(feature_names):
        ax.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale,
                 head_width=0.12, head_length=0.18, alpha=0.6)
        ax.text(loadings[i, 0] * scale * 1.08, loadings[i, 1] * scale * 1.08, feat, fontsize=7)

    ax.axhline(0, ls="--", lw=0.8); ax.axvline(0, ls="--", lw=0.8)
    ax.set_xlabel(f"Dim1 (PC1) — {expl_var[0]:.1f}%")
    ax.set_ylabel(f"Dim2 (PC2) — {expl_var[1]:.1f}%")
    ax.set_title("Biplot — Indivíduos e Variáveis")
    plt.tight_layout()
    plt.show()


def rank_features(
    X_df: pd.DataFrame,
    y: np.ndarray,
    contrib_df: pd.DataFrame,
    include: Iterable[str] = ("abs_corr_y", "mi", "contrib_PC1", "cos2_PC1"),
) -> pd.DataFrame:
    """
    Gera um ranking supervisionado combinando correlação com y, MI e métricas do PCA.

    Args:
        X_df: DataFrame de features.
        y: vetor binário 0/1 (1 = classe positiva).
        contrib_df: DataFrame com colunas 'contrib_PC1', 'contrib_PC2', 'cos2_PC1', 'cos2_PC2'.
        include: nomes dos critérios a combinar (entre 'abs_corr_y', 'mi', 'contrib_PC1',
                 'contrib_PC2', 'cos2_PC1', 'cos2_PC2').

    Returns:
        DataFrame indexado por feature, contendo colunas brutas e ranks normalizados (r_*),
        além de uma coluna 'score_composite' (média dos ranks escolhidos), ordenado do melhor
        para o pior.

    Notas:
        - Correlação usada: Pearson entre x (contínua) e y (binária).
        - MI: mutual_info_classif do scikit-learn (não-paramétrica).
    """
    def corr_with_y(col: pd.Series, y_arr: np.ndarray) -> float:
        x = col.to_numpy()
        if np.std(x) == 0:
            return 0.0
        return float(np.corrcoef(x, y_arr)[0, 1])

    abs_corr_y = X_df.apply(lambda c: abs(corr_with_y(c, y))).rename("abs_corr_y")
    mi = pd.Series(
        mutual_info_classif(X_df.to_numpy(), y, discrete_features=False, random_state=42),
        index=X_df.columns,
        name="mi",
    )

    df_rank = pd.concat([abs_corr_y, mi, contrib_df], axis=1)

    # Ranks normalizados (0–1): maior é melhor
    chosen = []
    for c in include:
        if c in df_rank.columns:
            r = df_rank[c].rank(ascending=False, method="average")
            df_rank[f"r_{c}"] = (r.max() - r) / (r.max() - 1.0)
            chosen.append(f"r_{c}")

    if not chosen:
        raise ValueError("Nenhuma métrica válida em 'include'.")

    df_rank["score_composite"] = df_rank[chosen].mean(axis=1)
    df_rank = df_rank.sort_values("score_composite", ascending=False)
    return df_rank


def prune_by_correlation(
    X_df: pd.DataFrame,
    priority: List[str],
    threshold: float = 0.90,
    method: str = "spearman",
) -> List[str]:
    """
    Poda gulosa por correlação: mantém apenas 1 de cada grupo com |ρ| ≥ limiar.

    Args:
        X_df: DataFrame de features.
        priority: lista de features por ordem de preferência (ex.: ranking supervisionado).
        threshold: limiar de correlação absoluta para considerar “redundante” (default: 0.90).
        method: método de correlação do pandas ('pearson', 'spearman', 'kendall').

    Returns:
        Lista de features selecionadas, mantendo a ordem de `priority`.

    Exemplo:
        >>> import pandas as pd, numpy as np
        >>> X = pd.DataFrame({"a":[1,2,3], "b":[2,4,6], "c":[1,1,0.9]})
        >>> prune_by_correlation(X, priority=["b","a","c"], threshold=0.95, method="pearson")
        ['b', 'c']
    """
    corr = X_df.corr(method=method).abs()
    kept: List[str] = []
    for feat in priority:
        if feat not in corr.columns:
            continue
        if all((corr.loc[feat, k] < threshold) for k in kept):
            kept.append(feat)
    return kept
