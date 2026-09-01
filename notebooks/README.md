# notebooks/

Ordem numerada obrigatória. Cada notebook roda de cima para baixo em ambiente limpo
(`Kernel → Restart & Run All`) sem erro — é isso que caracteriza reprodutibilidade.

| Arquivo | Escopo |
|---|---|
| `01_eda.ipynb` | distribuições, correlações, outliers, balanceamento |
| `02_preprocessamento.ipynb` | nulos, definição do alvo, normalização, features |
| `03_modelagem.ipynb` | split/CV, treino dos modelos, comparação |
| `04_avaliacao.ipynb` | métricas, feature importance, implicações de negócio |

## Convenções

- Numeração das células em ordem crescente (`[1]`, `[2]`, `[3]`...) — células fora de
  ordem indicam execução fora de sequência, o que compromete a reprodutibilidade.
- Saídas dos gráficos salvas no notebook, para que o resultado seja visível sem
  precisar executar nada.
- Todo gráfico tem um parágrafo de interpretação em markdown logo abaixo.
- A primeira célula de cada notebook importa `RANDOM_STATE` e os caminhos de
  [`src/config.py`](../src/config.py). O valor não muda entre notebooks.
