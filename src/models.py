"""Definição e treino dos modelos."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


def get_models() -> dict[str, Pipeline]:
    """Modelos candidatos, cada um encapsulado em um Pipeline.

    Usar Pipeline evita vazamento de dados: o scaler é ajustado apenas no fold
    de treino durante a validação cruzada.

    O alvo é desbalanceado (~14% de alta qualidade — ver notebooks/01_eda.ipynb,
    seção 5), por isso os classificadores que suportam ``class_weight`` usam
    ``"balanced"``: o custo de errar a classe minoritária passa a pesar mais no
    treino, em vez de o modelo simplesmente prever a classe majoritária sempre.
    """
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            )),
        ]),
    }
