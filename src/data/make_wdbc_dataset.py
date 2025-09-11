# src/data/make_wdbc_dataset.py
"""
Gera o CSV do Breast Cancer (sklearn) em data/wdbc.csv.

Uso (CLI):
    python -m src.data.make_wdbc_dataset                  # salva em data/wdbc.csv
    python -m src.data.make_wdbc_dataset --out data/meu_arquivo.csv

Também pode ser importado no notebook:
    from src.data.make_wdbc_dataset import build_wdbc
    build_wdbc(Path("data/wdbc.csv"))
"""
from pathlib import Path
import argparse
import pandas as pd
from sklearn.datasets import load_breast_cancer


def infer_repo_root(script_path: Path) -> Path:
    """Descobre a raiz do repositório partindo de src/ ou src/data/."""
    # .../src/data/arquivo.py -> raiz é parents[2]
    if script_path.parent.name == "data" and script_path.parent.parent.name == "src":
        return script_path.parents[2]
    # .../src/arquivo.py -> raiz é parents[1]
    if script_path.parent.name == "src":
        return script_path.parents[1]
    # fallback: um nível acima
    return script_path.parents[1]


def build_wdbc(out_path: Path) -> Path:
    """Gera o dataset WDBC (sklearn) e salva em out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer["data"], columns=[c.replace(" ", "_") for c in cancer["feature_names"]])

    target = pd.Categorical.from_codes(cancer["target"], cancer["target_names"])
    target = target.rename_categories({"malignant": "Maligno", "benign": "Benigno"})
    df["diagnosis"] = target.astype(str)

    df.to_csv(out_path, index=False)
    return out_path


def main(out: Path | None = None) -> Path:
    script_path = Path(__file__).resolve()
    repo_root = infer_repo_root(script_path)
    # default agora em data/wdbc.csv
    out_path = out or (repo_root / "data" / "raw" / "wdbc.csv")
    build_wdbc(out_path)
    print(f"[ok] CSV salvo em: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Caminho de saída (default: data/wdbc.csv)",
    )
    args = parser.parse_args()
    main(args.out)
