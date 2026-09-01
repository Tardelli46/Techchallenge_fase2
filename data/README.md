# data/

| Pasta | Conteúdo |
|---|---|
| `raw/` | `WineQT.csv` — arquivo original, exatamente como obtido da fonte, sem qualquer edição |
| `processed/` | `dataset_tratado.csv` — gerado pelo notebook `02_preprocessamento.ipynb` |

## Obtenção dos dados brutos

1. Baixar em: [Wine Quality Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
   — variante do Wine Quality Dataset publicado por Cortez *et al.* (2009), UCI Machine
   Learning Repository, restrita a vinhos tintos e acrescida da coluna `Id`.
2. Salvar como `data/raw/WineQT.csv`.

## Descrição dos arquivos

**`raw/WineQT.csv`** — 1.143 linhas × 13 colunas: 11 variáveis físico-químicas, a nota
de qualidade (`quality`, de 3 a 8) e o identificador `Id`. Não contém valores ausentes.

**`processed/dataset_tratado.csv`** — 1.143 linhas × 15 colunas. Em relação ao arquivo
bruto, a coluna `Id` é descartada e três colunas são acrescentadas: a variável resposta
binária `quality_bin` e os atributos derivados `acidity_ratio` e `alcohol_sulphates`.
A coluna `quality` é mantida apenas para rastreabilidade e **não** deve ser usada como
variável explicativa, uma vez que `quality_bin` deriva diretamente dela.
