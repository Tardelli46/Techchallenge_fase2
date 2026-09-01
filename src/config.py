"""Configuração central do projeto: caminhos, semente e constantes.

Importe daqui em todos os notebooks para que os resultados sejam reproduzíveis.
"""

from pathlib import Path

# --- Semente -----------------------------------------------------------------
# Use em TODO ponto que envolva aleatoriedade: train_test_split, modelos, CV.
RANDOM_STATE = 42

# --- Caminhos ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
METRICS = RESULTS / "metrics"

# --- Dataset -----------------------------------------------------------------
RAW_FILE = DATA_RAW / "WineQT.csv"
PROCESSED_FILE = DATA_PROCESSED / "dataset_tratado.csv"

ID_COL = "Id"              # coluna de índice do Kaggle, sem valor preditivo
TARGET = "quality"         # nota original (3-8), atribuída por especialistas
TARGET_BIN = "quality_bin"  # variável alvo do modelo: 1 = alta qualidade, 0 = baixa/média

# Limiar de binarização: nota >= 7 é tratada como "alta qualidade".
# Justificativa (ver notebooks/01_eda.ipynb, seção 5): no dataset, quality >= 7
# corresponde a 13,9% das amostras — um corte que separa o topo da distribuição
# (percentis 7 e 8, as notas mais raras e mais bem avaliadas) do restante,
# e reflete o critério informal do setor de tratar nota 7+ como "prêmio"/destaque.
QUALITY_THRESHOLD = 7

# --- Split -------------------------------------------------------------------
TEST_SIZE = 0.2
CV_FOLDS = 5
