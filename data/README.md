# data/

**Nada aqui é versionado.** O `.gitignore` bloqueia o conteúdo destas pastas de propósito:
datasets em Git incham o repositório e frequentemente violam a licença da fonte.

| Pasta | Conteúdo |
|---|---|
| `raw/` | arquivo original, exatamente como baixado da fonte — nunca editado |
| `processed/` | saída dos notebooks de pré-processamento (`.parquet` ou `.csv`) |

## Como obter

1. Baixe em: [Wine Quality Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
   (variante do Wine Quality Dataset original de Cortez et al., 2009, UCI Machine
   Learning Repository — apenas vinho tinto, com coluna `Id` adicional)
2. Salve como: `data/raw/WineQT.csv`
3. Checksum (opcional, recomendado): `shasum -a 256 data/raw/WineQT.csv`

Resumo: 1.143 linhas × 13 colunas (11 variáveis físico-químicas, `quality` e `Id`),
sem valores nulos.
