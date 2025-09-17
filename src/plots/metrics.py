# src/plots/metrics.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# seaborn é opcional; se não houver, caímos no matplotlib puro
try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

__all__ = [
    "evaluate_at_threshold",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curve",
]


def _operating_point(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:
    """
    Calcula, para um limiar arbitrário, o ponto ROC/PR:
    FPR, TPR (=Recall), Precision e Recall.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= thr).astype(int)

    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) else 0.0  # recall
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    zero_division: int = 0,
) -> Dict[str, Any]:
    """
    Calcula métricas de classificação para um dado limiar.

    Parâmetros
    ----------
    y_true : array-like de shape (n_samples,)
        Rótulos verdadeiros (0/1).
    y_score : array-like de shape (n_samples,)
        Probabilidades ou escores para a classe positiva.
    threshold : float, opcional (default=0.5)
        Limiar para converter probabilidade em classe predita (>= threshold -> 1).
    zero_division : {0,1}, opcional
        Como tratar divisão por zero em precisão/recall/F1 quando não há positivos
        previstos — repassado para métricas do scikit-learn.

    Retorna
    -------
    dict
        {"threshold", "roc_auc", "ap", "recall", "precision", "f1", "cm", "pred_pos", "pred_neg"}

    Exemplo
    -------
    >>> from src.plots.metrics import evaluate_at_threshold
    >>> info = evaluate_at_threshold(y_test, proba_test, threshold=0.53)
    >>> info["recall"], info["precision"], info["f1"]
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
        "recall": float(recall_score(y_true, y_pred, zero_division=zero_division)),
        "precision": float(precision_score(y_true, y_pred, zero_division=zero_division)),
        "f1": float(f1_score(y_true, y_pred, zero_division=zero_division)),
        "cm": cm,
        "pred_pos": int(y_pred.sum()),
        "pred_neg": int((1 - y_pred).sum()),
    }


def plot_confusion_matrix(
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    *,
    y_score: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    labels: Sequence[str] = ("Benigno", "Maligno"),
    normalize: Optional[str] = None,  # {"true","pred","all"} ou None
    title: str = "Matriz de Confusão",
    cmap: str = "Blues",
    use_seaborn: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plota a matriz de confusão. Aceita (y_true, y_pred) OU (y_true, y_score+threshold).

    Parâmetros
    ----------
    y_true : array-like (n amostras)
        Rótulos verdadeiros (0/1).
    y_pred : array-like (n amostras), opcional
        Predições binárias. Se não fornecido, é calculado a partir de y_score >= threshold.
    y_score : array-like (n amostras), opcional
        Probabilidade da classe positiva; usado apenas se y_pred for None.
    threshold : float, opcional
        Limiar para binarizar y_score.
    labels : sequência de 2 strings
        Nomes para eixo y (verdadeiro) e eixo x (predito), ordem [0, 1].
    normalize : {"true", "pred", "all"} ou None
        Normalização da matriz de confusão conforme scikit-learn.
    title : str
        Título do gráfico.
    cmap : str
        Mapa de cores para o heatmap.
    use_seaborn : bool
        Se True e seaborn disponível, usa sns.heatmap; caso contrário usa matplotlib.
    ax : matplotlib.axes.Axes, opcional
        Eixo no qual desenhar. Se None, cria uma nova figura.

    Retorna
    -------
    ax : matplotlib.axes.Axes

    Exemplos
    --------
    >>> # a partir de predição binária
    >>> from src.plots.metrics import plot_confusion_matrix
    >>> plot_confusion_matrix(y_true=y_test, y_pred=y_hat, labels=("Benigno","Maligno"))
    >>> plt.show()

    >>> # a partir de probabilidades e limiar
    >>> plot_confusion_matrix(y_true=y_test, y_score=proba_test, threshold=0.53)
    >>> plt.show()
    """
    if y_true is None:
        raise ValueError("y_true é obrigatório.")
    if y_pred is None:
        if y_score is None:
            raise ValueError("Forneça y_pred ou y_score.")
        y_pred = (np.asarray(y_score) >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if use_seaborn and _HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            xticklabels=list(labels),
            yticklabels=list(labels),
            square=True,
            cbar=True,
            ax=ax,
        )
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        # anotações
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )

    ax.set_title(title)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.tight_layout()
    return ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    title: str = "Curva ROC",
    mark_threshold: Optional[float] = None,
    exact_threshold: bool = False,
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Axes, Dict[str, Any]]:
    """
    Plota a Curva ROC com área sombreada e, opcionalmente, destaca um limiar.

    Parâmetros
    ----------
    y_true : array-like
        Rótulos verdadeiros (0/1).
    y_score : array-like
        Probabilidades/escores da classe positiva.
    title : str
        Título do gráfico.
    mark_threshold : float, opcional
        Se fornecido, marca no gráfico o ponto correspondente a esse limiar.
    exact_threshold : bool, default False
        - False: marca o **ponto da curva** cujo limiar interno é o mais próximo (com rótulo `thr≈`).
        - True : calcula o **ponto operacional exato** para `mark_threshold` (com rótulo `thr=`).
    color : str, opcional
        Cor da curva principal (matplotlib).
    ax : matplotlib.axes.Axes, opcional
        Eixo alvo. Se None, cria nova figura.

    Retorna
    -------
    (ax, info) : (matplotlib.axes.Axes, dict)
        info contém {"auc","fpr","tpr","marked_threshold","marked_point"}.

    Exemplo
    -------
    >>> from src.plots.metrics import plot_roc_curve
    >>> ax, info = plot_roc_curve(y_test, proba_test, mark_threshold=0.5)
    >>> plt.show()
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})", color=color)
    ax.fill_between(fpr, tpr, alpha=0.25, color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")

    # Marcar limiar específico, se solicitado
    if mark_threshold is not None:
        # thresholds em roc_curve mapeiam para pontos (fpr[i], tpr[i]).
        if exact_threshold:
            op = _operating_point(y_true, y_score, mark_threshold)
            ax.scatter([op["fpr"]], [op["tpr"]], s=60, marker="o", edgecolor="k",
                       label=f"thr={mark_threshold:.3f}")
            marked = {"fpr": op["fpr"], "tpr": op["tpr"], "thr": float(mark_threshold)}
        # Encontramos índice mais próximo do limiar.
        else:
            idx = int(np.argmin(np.abs(thr - mark_threshold)))
            ax.scatter([fpr[idx]], [tpr[idx]], s=60, marker="o", edgecolor="k",
                       label=f"thr≈{thr[idx]:.3f}")
            marked = {"fpr": float(fpr[idx]), "tpr": float(tpr[idx]), "thr": float(thr[idx])}
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Falsos Positivos (FPR)")
    ax.set_ylabel("Verdadeiros Positivos (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax, {
        "auc": float(roc_auc),
        "fpr": fpr,
        "tpr": tpr,
        "marked_threshold": float(mark_threshold) if mark_threshold is not None else None,
        "marked_point": marked,
    }


def _plot_iso_f1(ax: plt.Axes, f1_vals: Iterable[float] = (0.5, 0.6, 0.7, 0.8)) -> None:
    """Desenha linhas iso-F1 em um gráfico Precision–Recall."""
    recall = np.linspace(0.01, 1.0, 200)
    for f in f1_vals:
        prec = (f * recall) / (2 * recall - f + 1e-12)
        prec[(2 * recall - f) <= 0] = np.nan
        ax.plot(recall, prec, ls="--", lw=1, alpha=0.5, label=f"F1={f:.1f}")


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    title: str = "Curva Precision–Recall",
    mark_threshold: Optional[float] = None,
    exact_threshold: bool = False,
    show_iso_f1: bool = True,
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Axes, Dict[str, Any]]:
    """
    Plota a Curva Precision–Recall com área sombreada e, opcionalmente, linhas iso-F1.

    Parâmetros
    ----------
    y_true : array-like
        Rótulos verdadeiros (0/1).
    y_score : array-like
        Probabilidades/escores da classe positiva.
    title : str
        Título do gráfico.
    mark_threshold : float, opcional
        Se fornecido, marca no gráfico o ponto correspondente a esse limiar.
    exact_threshold : bool, default False
        - False: marca o **ponto da curva** cujo limiar é o mais próximo (rótulo `thr≈`).
        - True : calcula o **ponto operacional exato** para `mark_threshold` (rótulo `thr=`).
    show_iso_f1 : bool, opcional
        Desenha linhas iso-F1 para facilitar a leitura (default True).
    color : str, opcional
        Cor da curva principal (matplotlib).
    ax : matplotlib.axes.Axes, opcional
        Eixo alvo. Se None, cria nova figura.

    Retorna
    -------
    (ax, info) : (matplotlib.axes.Axes, dict)
        info contém {"ap","precision","recall","thresholds","marked_point"}.

    Exemplo
    -------
    >>> from src.plots.metrics import plot_pr_curve
    >>> ax, info = plot_pr_curve(y_test, proba_test, mark_threshold=0.5, show_iso_f1=True)
    >>> plt.show()
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(rec, prec, lw=2, label=f"PR (AP = {ap:.4f})", color=color)
    ax.fill_between(rec, prec, alpha=0.25, color=color)

    if show_iso_f1:
        _plot_iso_f1(ax)

    # Marcar limiar específico, se solicitado
        if exact_threshold:
            op = _operating_point(y_true, y_score, mark_threshold)
            ax.scatter([op["recall"]], [op["precision"]], s=60, marker="o", edgecolor="k",
                       label=f"thr={mark_threshold:.3f}")
            marked = {"recall": op["recall"], "precision": op["precision"], "thr": float(mark_threshold)}
        elif len(thr) > 0:
            idx = int(np.argmin(np.abs(thr - mark_threshold)))
            # precision_recall_curve retorna prec/rec com len(thr)+1
            ax.scatter([rec[idx + 1]], [prec[idx + 1]], s=60, marker="o", edgecolor="k",
                       label=f"thr≈{thr[idx]:.3f}")
            marked = {"recall": float(rec[idx + 1]), "precision": float(prec[idx + 1]), "thr": float(thr[idx])}

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax, {
        "ap": float(ap),
        "precision": prec,
        "recall": rec,
        "thresholds": thr,
        "marked_point": marked,
    }

