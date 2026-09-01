"""Limpeza, escala e feature engineering.

Se você decidir NÃO criar features novas, documente a decisão e o porquê —
a justificativa da não-aplicação vale tanto quanto a aplicação.
"""

import pandas as pd


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo de nulos por coluna, em contagem e percentual."""
    total = df.isna().sum()
    pct = (total / len(df) * 100).round(2)
    return (
        pd.DataFrame({"nulos": total, "pct": pct})
        .query("nulos > 0")
        .sort_values("nulos", ascending=False)
    )


def binarize_target(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    """Converte uma variável contínua em binária a partir de um limiar.

    O limiar é uma decisão de modelagem: justifique-o no README com base na
    distribuição observada, não apenas por convenção.
    """
    out = df.copy()
    out[f"{column}_bin"] = (out[column] >= threshold).astype(int)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as duas features derivadas usadas no projeto.

    Ambas foram escolhidas por terem correlação com o alvo mais forte do que
    qualquer uma das colunas originais que as compõem isoladamente (checado em
    notebooks/02_preprocessamento.ipynb, seção 4):

    - ``acidity_ratio``: acidez fixa / acidez volátil. Acidez fixa dá frescor,
      acidez volátil (ácido acético) causa gosto de vinagre em excesso — a razão
      entre as duas captura esse equilíbrio melhor do que cada termo sozinho.
    - ``alcohol_sulphates``: interação entre as duas variáveis com maior
      correlação positiva com a qualidade (teor alcoólico e sulfatos), para dar
      a modelos lineares acesso a esse efeito conjunto sem precisar de termos
      polinomiais explícitos.
    """
    out = df.copy()
    out["acidity_ratio"] = out["fixed acidity"] / out["volatile acidity"]
    out["alcohol_sulphates"] = out["alcohol"] * out["sulphates"]
    return out
