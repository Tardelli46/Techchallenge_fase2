"""Métricas, gráficos de avaliação e comparação entre modelos."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import METRICS


def score(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Conjunto de métricas para classificação binária.

    Acurácia sozinha engana em base desbalanceada — por isso precisão, recall,
    F1 e AUC vêm junto.
    """
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    return out


def comparison_table(results: dict[str, dict], filename: str = "comparacao_modelos.csv") -> pd.DataFrame:
    """Monta a tabela comparativa e salva em results/metrics/."""
    df = pd.DataFrame(results).T.round(4).sort_values("f1", ascending=False)
    METRICS.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS / filename)
    return df


def plot_confusion_matrices(models: dict, X_test, y_test, path=None):
    """Matriz de confusão de cada modelo, lado a lado, no mesmo eixo de cores."""
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2))
    if len(models) == 1:
        axes = [axes]
    for ax, (nome, modelo) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_estimator(
            modelo, X_test, y_test, ax=ax, colorbar=False,
            display_labels=["Baixa/Média", "Alta"],
        )
        ax.set_title(nome)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(models: dict, X_test, y_test, path=None):
    """Curvas ROC de todos os modelos sobrepostas, para comparar diretamente."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for nome, modelo in models.items():
        RocCurveDisplay.from_estimator(modelo, X_test, y_test, ax=ax, name=nome)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Aleatório")
    ax.legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def feature_importance(modelo, feature_names) -> pd.Series:
    """Extrai importância de variáveis de um Pipeline treinado.

    Usa ``feature_importances_`` (árvores) ou o valor absoluto de ``coef_``
    (modelos lineares) — o que existir no classificador final do pipeline.
    """
    clf = modelo.named_steps["clf"] if hasattr(modelo, "named_steps") else modelo
    if hasattr(clf, "feature_importances_"):
        valores = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        valores = abs(clf.coef_[0])
    else:
        raise AttributeError(f"{clf} não expõe feature_importances_ nem coef_.")
    return pd.Series(valores, index=feature_names).sort_values(ascending=False)
