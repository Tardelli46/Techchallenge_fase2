# data/

Os conjuntos de dados **não são versionados**: o `.gitignore` bloqueia o conteúdo destas
pastas, mantendo no repositório apenas este arquivo e os marcadores `.gitkeep`. A opção
evita inflar o repositório e respeita a licença da fonte original.

| Pasta | Conteúdo |
|---|---|
| `raw/` | `WineQT.csv` — arquivo original, exatamente como obtido da fonte, sem qualquer edição |
| `processed/` | `dataset_tratado.csv` — gerado pelo notebook `02_preprocessamento.ipynb` |

## Obtenção dos dados brutos

1. Baixar em: [Wine Quality Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
   — variante do Wine Quality Dataset publicado por Cortez *et al.* (2009), UCI Machine
   Learning Repository, restrita a vinhos tintos e acrescida da coluna `Id`.
2. Salvar como `data/raw/WineQT.csv`.
3. Conferir a integridade do arquivo (opcional):

```powershell
Get-FileHash data\raw\WineQT.csv -Algorithm SHA256    # Windows
sha256sum data/raw/WineQT.csv                          # Linux
shasum -a 256 data/raw/WineQT.csv                      # macOS
```

SHA-256 esperado:
`7e38cc28812d08f521ee19e29e9d3622cde03464ff5e9a8b14aa991ec74ae49e`

## Descrição dos arquivos

**`raw/WineQT.csv`** — 1.143 linhas × 13 colunas: 11 variáveis físico-químicas, a nota
de qualidade (`quality`, de 3 a 8) e o identificador `Id`. Não contém valores ausentes.

**`processed/dataset_tratado.csv`** — 1.143 linhas × 15 colunas. Em relação ao arquivo
bruto, a coluna `Id` é descartada e três colunas são acrescentadas: a variável resposta
binária `quality_bin` e os atributos derivados `acidity_ratio` e `alcohol_sulphates`.
A coluna `quality` é mantida apenas para rastreabilidade e **não** deve ser usada como
variável explicativa, uma vez que `quality_bin` deriva diretamente dela.
