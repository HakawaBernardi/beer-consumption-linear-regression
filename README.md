# 🍺 Regressão Linear — Consumo de Cerveja

> **Disciplina:** Inteligência Artificial — PUC Goiás  
> **Aluno:** Hakawã Luiz Bernardi

---

## 📋 Descrição

Desenvolvimento de um modelo de **Regressão Linear Múltipla** para predição do consumo de cerveja em função da temperatura, utilizando o dataset `beer_consuption.csv`.

O projeto implementa dois métodos de estimação dos coeficientes **do zero**, sem uso de funções prontas de ML:

| Método | Abordagem |
|---|---|
| **MMQ** | Mínimos Quadrados Ordinários — solução analítica via β̂ = (XᵀX)⁻¹Xᵀy |
| **Gradiente Descendente** | Otimização iterativa com atualização θ := θ − α∇J(θ) |

---

## 🚀 Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/HakawaBernardi/beer-consumption-linear-regression.git
cd beer-consumption-linear-regression
```

### 2. Instale as dependências

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Execute o script

```bash
python regressao_cerveja.py
```

---

## 📊 Dataset

**beer_consuption.csv** — Dados diários de consumo de cerveja

| Coluna | Tipo | Descrição |
|---|---|---|
| Data | object | Data da observação |
| Temperatura Media (C) | float64 | Temperatura média diária |
| Temperatura Minima (C) | float64 | Temperatura mínima diária |
| Temperatura Maxima (C) | float64 | Temperatura máxima diária |
| Precipitacao (mm) | float64 | Precipitação diária |
| Final de Semana | int64 | 1 = fim de semana, 0 = dia útil |
| Consumo de cerveja (litros) | float64 | **Variável alvo** |

- **365 observações × 7 variáveis** — sem valores faltantes

---

## 🔬 O que o Script Faz

### Análise Exploratória (EDA)
- `head()` / `tail()` / `shape` / `dtypes` / `isna()`
- Estatísticas descritivas (`describe`)
- Correlação de Pearson e Spearman
- Gráficos: barras, linhas, precipitação, consumo, correlograma, boxplots, histogramas, dispersão

### Modelos de Regressão Linear Múltipla

**Features utilizadas:** Temperatura Média, Mínima, Máxima, Precipitação e Final de Semana.

#### Resultados obtidos

| Coeficiente | MMQ | Grad. Descendente |
|---|---|---|
| β₀ (intercepto) | 6.4447 | 6.45 |
| β₁ (Temp. Média) | 0.0308 | 0.031 |
| β₂ (Temp. Mínima) | −0.0190 | −0.019 |
| β₃ (Temp. Máxima) | 0.6560 | 0.655 |
| β₄ (Precipitação) | −0.0575 | −0.058 |
| β₅ (Final Semana) | 5.1832 | 5.18 |

### Métricas de Avaliação

| Métrica | MMQ | Grad. Descendente |
|---|---|---|
| R² | 0.75 | 0.75 |
| MSE | 10.25 | 10.30 |
| RMSE | 3.20 | 3.21 |
| MAE | 2.50 | 2.51 |

> Os dois métodos convergem para coeficientes praticamente idênticos, validando ambas as implementações.

---

## 🛠️ Tecnologias

- Python 3.x · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn

---

## 📄 Licença

Projeto acadêmico — PUC Goiás, 2025.