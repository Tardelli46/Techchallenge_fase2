# Classificação da Qualidade de Vinhos com Machine Learning

> **Tech Challenge — Fase 2 · POSTECH Data Analytics**
> Turma 14DTAT · Grupo 23

![Python](https://img.shields.io/badge/Linguagem-Python-blue)
![Jupyter](https://img.shields.io/badge/Ambiente-Jupyter%20Notebook-orange)
![Scikit-learn](https://img.shields.io/badge/Biblioteca-Scikit--learn-red)
![Machine Learning](https://img.shields.io/badge/Tema-Classificação%20Binária-purple)
![Status](https://img.shields.io/badge/Status-Concluído-green)

---

## Sumário

1. [Introdução](#1-introdução)
2. [Objetivos](#2-objetivos)
3. [Materiais e métodos](#3-materiais-e-métodos)
4. [Resultados](#4-resultados)
5. [Discussão](#5-discussão)
6. [Limitações e trabalhos futuros](#6-limitações-e-trabalhos-futuros)
7. [Reprodutibilidade](#7-reprodutibilidade)
8. [Organização do repositório](#8-organização-do-repositório)
9. [Entregas do desafio](#9-entregas-do-desafio)
10. [Referências](#10-referências)

---

## 1. Introdução

A avaliação da qualidade de um vinho é tradicionalmente realizada por especialistas,
por meio de análise sensorial que considera aroma, sabor, acidez e equilíbrio. Embora
consolidado, esse procedimento apresenta limitações relevantes: é subjetivo, demanda
tempo e depende diretamente da experiência do avaliador, o que dificulta sua aplicação
em escala.

Durante o processo produtivo, contudo, são registradas diversas medições
físico-químicas do produto — acidez, teor alcoólico, densidade, concentração de dióxido
de enxofre, entre outras — obtidas de forma mais rápida e menos onerosa do que uma
degustação formal. O presente trabalho investiga em que medida essas variáveis
laboratoriais permitem antecipar a classificação de qualidade atribuída posteriormente
por especialistas.

A aplicação prática pretendida é a triagem: um modelo preditivo capaz de indicar os
lotes com maior probabilidade de obter avaliação elevada permite direcionar o tempo do
enólogo aos casos mais promissores, sem substituir a análise sensorial, que permanece
como fonte da nota efetiva.

---

## 2. Objetivos

### 2.1 Objetivo geral

Desenvolver e avaliar modelos de classificação capazes de prever, a partir de
características físico-químicas, se um vinho será classificado como de alta qualidade.

### 2.2 Objetivos específicos

* Caracterizar o comportamento das variáveis físico-químicas por meio de análise
  exploratória;
* Transformar a variável de qualidade em uma classificação binária, com limiar
  justificado pela distribuição observada;
* Tratar os dados e avaliar a criação de variáveis derivadas;
* Treinar e comparar diferentes algoritmos de classificação;
* Avaliar o desempenho dos modelos com métricas adequadas ao desbalanceamento das
  classes;
* Identificar as variáveis de maior influência sobre a qualidade;
* Traduzir os resultados obtidos em implicações para o processo produtivo.

---

## 3. Materiais e métodos

### 3.1 Base de dados

Foi utilizado o **Wine Quality Dataset**, disponível publicamente no Kaggle, derivado
do conjunto de dados originalmente publicado por Cortez *et al.* (2009). O arquivo
`WineQT.csv` reúne **1.143 amostras de vinho tinto**, descritas por 11 variáveis
físico-químicas e por uma nota de qualidade atribuída por especialistas. Não foram
identificados valores ausentes.

🔗 [Wine Quality Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

| Variável | Descrição |
|---|---|
| `fixed acidity` | Acidez fixa, referente aos ácidos não voláteis |
| `volatile acidity` | Acidez volátil (ácido acético); em excesso, associada a defeito sensorial |
| `citric acid` | Ácido cítrico, relacionado ao frescor e ao equilíbrio |
| `residual sugar` | Açúcar residual remanescente após a fermentação |
| `chlorides` | Concentração de cloretos |
| `free sulfur dioxide` | Dióxido de enxofre livre, com ação antioxidante e antimicrobiana |
| `total sulfur dioxide` | Dióxido de enxofre total (livre e ligado) |
| `density` | Densidade, influenciada pelos teores de açúcar e álcool |
| `pH` | Medida de acidez ou basicidade |
| `sulphates` | Sulfatos, associados à conservação e à estabilidade |
| `alcohol` | Teor alcoólico |
| `quality` | Nota de qualidade original, de 3 a 8 |
| `Id` | Identificador do registro; descartado por não possuir valor preditivo |

### 3.2 Definição da variável resposta

A variável `quality`, originalmente ordinal, foi convertida em uma variável binária
(`quality_bin`), conforme o critério:

```text
quality >= 7  →  1  (Alta Qualidade)
quality <  7  →  0  (Baixa/Média Qualidade)
```

A definição do limiar foi submetida a verificação empírica, comparando-se três cortes
possíveis contra a distribuição observada. O corte em 6 classificaria 54,33% da base
como alta qualidade, resultando em segmentação pouco informativa; o corte em 8
retornaria apenas 1,40% de casos positivos, quantidade insuficiente para treinamento
estável. O limiar adotado (7) resulta em **13,91% de observações positivas** — 159 das
1.143 amostras —, proporção compatível com a delimitação de um segmento superior e
ainda suficiente para as etapas de treino e validação.

### 3.3 Pré-processamento e engenharia de atributos

Verificada a ausência de valores nulos, a padronização das variáveis foi realizada por
meio de `StandardScaler`, escolhido em detrimento da normalização min-max em razão da
presença de valores extremos legítimos em variáveis como `chlorides` e `residual
sugar`. O ajuste do escalonador foi encapsulado em `Pipeline`, de modo que ocorra
exclusivamente sobre a partição de treino — procedimento que evita o vazamento de
informação do conjunto de teste.

Foram avaliadas diversas combinações de variáveis, das quais duas foram mantidas por
apresentarem correlação com a variável resposta superior à das colunas que as originam:

| Atributo derivado | Definição | Justificativa |
|---|---|---|
| `acidity_ratio` | `fixed acidity / volatile acidity` | Correlação de 0,32 com a variável resposta, superior à de `fixed acidity` isolada (0,12); expressa o equilíbrio entre frescor e defeito sensorial |
| `alcohol_sulphates` | `alcohol × sulphates` | Disponibiliza a interação entre as duas variáveis de maior correlação positiva individual, informação não capturada por modelos lineares sem termo explícito |

### 3.4 Modelagem e validação

A base foi dividida em treino e teste na proporção 80/20, com estratificação pela
variável resposta, preservando a proporção original de classes em ambas as partições.
Foram treinados três classificadores distintos:

1. **Regressão Logística** — modelo linear de referência, com coeficientes
   interpretáveis;
2. **Random Forest** — ensemble de árvores, capaz de representar relações não lineares
   e interações;
3. **Gradient Boosting** — ensemble sequencial, usualmente competitivo em bases de
   pequeno porte.

Adotou-se validação cruzada estratificada de cinco partições sobre o conjunto de
treino, com o parâmetro `class_weight="balanced"` nos algoritmos que o suportam, de
modo a compensar o desbalanceamento das classes. A semente aleatória foi fixada em 42
em todos os pontos que envolvem aleatoriedade, garantindo a reprodutibilidade dos
resultados.

---

## 4. Resultados

### 4.1 Desempenho comparativo

Métricas obtidas sobre o conjunto de teste (229 observações), não utilizado em nenhuma
etapa de treinamento ou validação:

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **Random Forest** | 0,8952 | 0,6111 | 0,6875 | **0,6471** | **0,9006** |
| Gradient Boosting | 0,9039 | 0,6923 | 0,5625 | 0,6207 | 0,8683 |
| Regressão Logística | 0,7991 | 0,3793 | 0,6875 | 0,4889 | 0,8499 |

Considerando o desbalanceamento da variável resposta, a acurácia isolada mostra-se
insuficiente como critério de seleção: um classificador que atribuísse a todos os casos
a classe majoritária alcançaria 86% de acurácia sem identificar corretamente nenhum
vinho de alta qualidade. Por essa razão, a comparação foi conduzida com base no
F1-Score, que equilibra precisão e revocação, e na AUC-ROC, que mensura a capacidade
discriminatória independentemente do limiar de decisão.

O **Random Forest** apresentou o melhor desempenho nas duas métricas prioritárias,
superando o Gradient Boosting por margem estreita e a Regressão Logística por margem
expressiva. O desempenho inferior do modelo linear sugere que a relação entre as
variáveis físico-químicas e a qualidade não é adequadamente descrita por uma fronteira
linear.

### 4.2 Variáveis de maior influência

A análise de importância das variáveis no modelo selecionado indica a seguinte ordem de
relevância:

| Posição | Variável | Importância |
|---|---|---|
| 1º | `alcohol_sulphates` | 0,1599 |
| 2º | `alcohol` | 0,1501 |
| 3º | `acidity_ratio` | 0,1115 |
| 4º | `citric acid` | 0,0916 |
| 5º | `sulphates` | 0,0837 |
| 6º | `volatile acidity` | 0,0810 |

Os dois atributos derivados na etapa de pré-processamento ocupam a primeira e a
terceira posições, e as três variáveis de maior peso concentram aproximadamente 40% da
importância total atribuída pelo modelo.

---

## 5. Discussão

Os resultados obtidos convergem com o conhecimento consolidado sobre a enologia. O
**teor alcoólico** apresenta a associação positiva mais forte com a qualidade, ao passo
que a **acidez volátil** constitui o principal fator negativo — comportamento esperado,
uma vez que concentrações elevadas de ácido acético são percebidas como defeito
sensorial. Os **sulfatos** e o **ácido cítrico**, associados respectivamente à
conservação e ao frescor do produto, também figuram entre as variáveis relevantes.

Sob a perspectiva do processo produtivo, três frentes de monitoramento se destacam: o
controle da fermentação, responsável pelo teor alcoólico final; a dosagem de sulfatos,
empregada como conservante e antioxidante; e a prevenção da acidez volátil elevada,
usualmente indicativa de contaminação microbiológica ou oxidação, e não de
característica varietal.

Quanto ao emprego operacional, o modelo selecionado identifica aproximadamente sete em
cada dez vinhos efetivamente classificados como de alta qualidade. Esse patamar é
compatível com o uso pretendido de triagem — priorização da agenda de avaliação
sensorial —, mas não com a substituição do julgamento especializado. Cabe registrar que
a definição do limiar de decisão deve considerar o custo relativo dos erros: em um
cenário de triagem, o falso negativo, que descarta um lote efetivamente promissor,
tende a ser mais oneroso do que o falso positivo, que apenas gera reavaliação
adicional.

---

## 6. Limitações e trabalhos futuros

O conjunto de dados utilizado é de dimensão reduzida (1.143 observações) e restrito a
vinhos tintos, não contemplando informações de safra, casta ou região, o que limita a
generalização dos resultados. A classe correspondente à nota 8 conta com apenas 16
registros, reduzindo a confiabilidade das previsões no extremo superior da escala.
Adicionalmente, a variável resposta constitui um julgamento humano, sujeito a variação
entre avaliadores, de forma que o modelo reproduz uma percepção agregada e não uma
medida objetiva de qualidade.

Como desdobramentos, recomenda-se:

* avaliar técnicas de balanceamento, como SMOTE, em complemento ao uso de pesos de classe;
* comparar o desempenho com algoritmos de boosting adicionais, como XGBoost e LightGBM;
* calibrar o limiar de decisão a partir do custo real associado a cada tipo de erro;
* validar o modelo em safras não representadas na amostra de treinamento;
* estender a análise a vinhos brancos e à base combinada.

---

## 7. Reprodutibilidade

```bash
git clone https://github.com/Tardelli46/Techchallenge_fase2.git
cd Techchallenge_fase2

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

O conjunto de dados deve ser obtido no [Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
e armazenado como `data/raw/WineQT.csv`, uma vez que os dados não são versionados
(ver [`data/README.md`](data/README.md)). A execução dos notebooks deve seguir a ordem
numerada:

| Ordem | Notebook | Conteúdo |
|---|---|---|
| 1 | [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Análise exploratória dos dados |
| 2 | [`02_preprocessamento.ipynb`](notebooks/02_preprocessamento.ipynb) | Tratamento, binarização e engenharia de atributos |
| 3 | [`03_modelagem.ipynb`](notebooks/03_modelagem.ipynb) | Particionamento, validação cruzada e treinamento |
| 4 | [`04_avaliacao.ipynb`](notebooks/04_avaliacao.ipynb) | Métricas, interpretação e conclusões |

A constante `RANDOM_STATE = 42`, definida em [`src/config.py`](src/config.py) e
importada por todos os notebooks, assegura que a execução sequencial em ambiente limpo
reproduza integralmente os resultados apresentados na Seção 4.

---

## 8. Organização do repositório

```text
├── data/
│   ├── raw/                            conjunto de dados original (não versionado)
│   └── processed/                      base tratada, gerada pelo notebook 02
├── notebooks/
│   ├── 01_eda.ipynb                    análise exploratória
│   ├── 02_preprocessamento.ipynb       tratamento e engenharia de atributos
│   ├── 03_modelagem.ipynb              treinamento e validação
│   └── 04_avaliacao.ipynb              avaliação e interpretação
├── src/
│   ├── config.py                       caminhos, semente e constantes
│   ├── data.py                         carregamento e persistência
│   ├── preprocessing.py                tratamento e atributos derivados
│   ├── models.py                       definição dos modelos
│   └── evaluation.py                   métricas e gráficos de avaliação
├── results/
│   ├── figures/                        gráficos exportados
│   ├── metrics/                        tabelas comparativas
│   └── models/                         modelos treinados (não versionados)
├── docs/
│   └── apresentacao_executiva.pdf      apresentação executiva
├── requirements.txt
└── README.md
```

---

## 9. Entregas do desafio

| Item | Endereço |
|---|---|
| Repositório | https://github.com/Tardelli46/Techchallenge_fase2 |
| Apresentação executiva | [`docs/apresentacao_executiva.pdf`](docs/apresentacao_executiva.pdf) |

### Tecnologias utilizadas

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter Notebook ·
Joblib. As versões utilizadas encontram-se fixadas em
[`requirements.txt`](requirements.txt).

---

## 10. Referências

CORTEZ, P.; CERDEIRA, A.; ALMEIDA, F.; MATOS, T.; REIS, J. Modeling wine preferences by
data mining from physicochemical properties. **Decision Support Systems**, v. 47, n. 4,
p. 547-553, 2009.

PEDREGOSA, F. *et al.* Scikit-learn: Machine Learning in Python. **Journal of Machine
Learning Research**, v. 12, p. 2825-2830, 2011.

WINE QUALITY DATASET. **Kaggle**, 2022. Disponível em:
https://www.kaggle.com/datasets/yasserh/wine-quality-dataset. Acesso em: 28 ago. 2026.

---

## Autoria

| Nome | RM |
|---|---|
| Felipe Lins | 374386 |
