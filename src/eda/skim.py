# src/eda/skim.py
"""
Utilitários de EDA: resumos numéricos e categóricos com histograma ASCII.

Exemplo rápido
--------------
>>> import pandas as pd
>>> from src.eda.skim import skim_numeric, skim_categorical
>>> df = pd.read_csv("data/raw/wdbc.csv")              # dataset do projeto
>>> skim_numeric(df).head(3)       # doctest: +SKIP
>>> skim_categorical(df)           # doctest: +SKIP
"""

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["skim_numeric", "skim_categorical"]

def _histograma_ascii(dados, bins=5):
    """Cria um histograma ASCII simples."""
    counts, _ = np.histogram(dados, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1
    chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    hist = ""
    for c in counts:
        idx = int((c / max_count) * (len(chars) - 1)) if max_count else 0
        hist += chars[min(idx, len(chars) - 1)]
    return hist

def skim_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera um resumo estatístico para colunas numéricas.

    Args:
        df: DataFrame de entrada; apenas colunas numéricas são consideradas.

    Returns:
        DataFrame com as colunas:
        ['variable','miss','p_zeros','unique','mean','sd','CV',
         'p0','p25','p50','p75','p100','IQR','hist','skew','kurt'].

    Example:
        >>> import pandas as pd, numpy as np
        >>> X = pd.DataFrame({'a':[0,1,2,np.nan], 'b':[5,5,5,5]})
        >>> skim_numeric(X)  # doctest: +SKIP
    """
    df_num = df.select_dtypes(include=[np.number])
    resultados = []
    for col in df_num.columns:
        dados_full = df_num[col]
        dados = dados_full.dropna()

        miss = dados_full.isnull().sum()
        mean_val = dados.mean()
        sd_val = dados.std()
        p0 = dados.min()
        p25 = dados.quantile(0.25)
        p50 = dados.quantile(0.50)
        p75 = dados.quantile(0.75)
        p100 = dados.max()

        q_zeros = (dados == 0).sum()
        p_zeros = (q_zeros / len(dados)) if len(dados) > 0 else 0

        cv = (sd_val / mean_val) if mean_val != 0 else np.nan
        iqr = p75 - p25

        skew_val = stats.skew(dados) if len(dados) > 1 else np.nan
        kurt_val = stats.kurtosis(dados) if len(dados) > 1 else np.nan  # excesso

        unique_val = dados.nunique()
        hist_ascii = _histograma_ascii(dados)

        resultados.append({
            "variable": col,
            "miss": miss,
            "p_zeros": round(p_zeros, 4),
            "unique": unique_val,
            "mean": round(mean_val, 3),
            "sd": round(sd_val, 3),
            "CV": round(cv, 3),
            "p0": round(p0, 3),
            "p25": round(p25, 3),
            "p50": round(p50, 3),
            "p75": round(p75, 3),
            "p100": round(p100, 3),
            "IQR": round(iqr, 3),
            "hist": hist_ascii,
            "skew": round(skew_val, 3),
            "kurt": round(kurt_val, 3)
        })
    return pd.DataFrame(resultados)

def skim_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera um resumo para colunas categóricas (object/category).

    Args:
        df: DataFrame de entrada; apenas colunas categóricas são consideradas.

    Returns:
        DataFrame com as colunas:
        ['variable','miss','ordered','n_unique','top_counts'].

    Example:
        >>> import pandas as pd
        >>> X = pd.DataFrame({'y':['a','a','b',None]})
        >>> skim_categorical(X)  # doctest: +SKIP
    """
    df_cat = df.select_dtypes(include=["object", "category"])
    resultados = []
    for col in df_cat.columns:
        dados_full = df_cat[col]
        dados = dados_full.dropna()

        miss = dados_full.isnull().sum()
        ordered = hasattr(dados.dtype, "ordered") and getattr(dados.dtype, "ordered")
        n_unique = dados.nunique()

        vc = dados.value_counts().head(5)
        top_counts = ", ".join([f"{k[:4]}: {v}" for k, v in vc.items()])

        resultados.append({
            "variable": col,
            "miss": miss,
            "ordered": ordered,
            "n_unique": n_unique,
            "top_counts": top_counts
        })
    return pd.DataFrame(resultados)
