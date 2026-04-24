# Importação das bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Leitura do arquivo de dados
cerveja = pd.read_csv('beer_consuption.csv')

# 2. Primeiras observações do arquivo
print("Primeiras observações:")
print(cerveja.head())

# 3. Últimas observações do arquivo
print("\nÚltimas observações:")
print(cerveja.tail())

# 4. Dimensão da base de dados
print("\nDimensão da base de dados:")
print(cerveja.shape)

# 5. Verificação de valores faltantes
print("\nValores faltantes:")
print(cerveja.isna().sum())

# 6. Verificação dos tipos das variáveis
print("\nTipos das variáveis:")
print(cerveja.dtypes)

# 7. Correlação entre as variáveis
# Remover a coluna 'Data' antes de calcular a correlação
cerveja_numerica = cerveja.drop(columns=['Data'])
print("\nCorrelação entre as variáveis:")
correlacao = cerveja_numerica.corr()
print(correlacao)

# 8. Tabela descritiva das variáveis
print("\nTabela descritiva das variáveis:")
print(cerveja.describe())

# 9. Visualização de dados

# a. Gráfico de barras para Final de Semana
plt.figure(figsize=(6, 4))
sns.countplot(x='Final de Semana', data=cerveja)
plt.title('Distribuição de Final de Semana')
plt.show()

# b. Gráfico das temperaturas média, mínima e máxima
plt.figure(figsize=(10, 6))
plt.plot(cerveja['Temperatura Media (C)'], label='Temperatura Média')
plt.plot(cerveja['Temperatura Minima (C)'], label='Temperatura Mínima')
plt.plot(cerveja['Temperatura Maxima (C)'], label='Temperatura Máxima')
plt.legend()
plt.title('Temperaturas ao Longo do Tempo')
plt.show()

# c. Gráfico da precipitação diária
plt.figure(figsize=(10, 6))
plt.plot(cerveja['Precipitacao (mm)'])
plt.title('Precipitação Diária')
plt.show()

# d. Gráfico do consumo de cerveja
plt.figure(figsize=(10, 6))
plt.plot(cerveja['Consumo de cerveja (litros)'])
plt.title('Consumo de Cerveja ao Longo do Tempo')
plt.show()

# e. Correlograma com correlação de Pearson
plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, annot=True, cmap='coolwarm')
plt.title('Correlograma de Pearson')
plt.show()

# f. Boxplots para verificação de outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=cerveja[['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)', 'Precipitacao (mm)', 'Consumo de cerveja (litros)']])
plt.title('Boxplots das Variáveis')
plt.show()

# g. Histograma das variáveis
cerveja[['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)', 'Precipitacao (mm)', 'Consumo de cerveja (litros)']].hist(bins=20, figsize=(15, 10))
plt.suptitle('Histograma das Variáveis')
plt.show()

# h. Gráfico de dispersão entre as variáveis
sns.pairplot(cerveja[['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)', 'Precipitacao (mm)', 'Consumo de cerveja (litros)']])
plt.suptitle('Gráfico de Dispersão entre as Variáveis')
plt.show()

# 10. construcao do modelo de regressao linear
X = cerveja[['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)', 'Precipitacao (mm)', 'Final de Semana']]
y = cerveja['Consumo de cerveja (litros)']

X = np.c_[np.ones(X.shape[0]), X]

# a. Método dos Mínimos Quadrados (MMQ)
coeficientes_mmq = np.linalg.inv(X.T @ X) @ X.T @ y
print("\nCoeficientes MMQ:", coeficientes_mmq)

# b. Método do Gradiente Descendente
theta = np.zeros(X.shape[1])  # inicializacao dos parametros
alpha = 0.01  # taxa de aprendizado
iterations = 1000  # numero de iteracoes

for _ in range(iterations):
    gradients = (2/X.shape[0]) * X.T @ (X @ theta - y)
    theta -= alpha * gradients

print("\nCoeficientes Gradiente Descendente:", theta)

# c. Comparação dos resultados
print("\nComparação dos coeficientes:")
print("Coeficientes MMQ:", coeficientes_mmq)
print("Coeficientes Gradiente Descendente:", theta)

# 11. Cálculo das métricas de avaliação
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Previsões do modelo MMQ
y_pred_mmq = X @ coeficientes_mmq

# Previsões do modelo Gradiente Descendente
y_pred_gd = X @ theta

# Métricas para MMQ
r2_mmq = r2_score(y, y_pred_mmq)
mse_mmq = mean_squared_error(y, y_pred_mmq)
rmse_mmq = np.sqrt(mse_mmq)
mae_mmq = mean_absolute_error(y, y_pred_mmq)

# Métricas para Gradiente Descendente
r2_gd = r2_score(y, y_pred_gd)
mse_gd = mean_squared_error(y, y_pred_gd)
rmse_gd = np.sqrt(mse_gd)
mae_gd = mean_absolute_error(y, y_pred_gd)

print("\nMétricas MMQ:")
print(f"R²: {r2_mmq}, MSE: {mse_mmq}, RMSE: {rmse_mmq}, MAE: {mae_mmq}")

print("\nMétricas Gradiente Descendente:")
print(f"R²: {r2_gd}, MSE: {mse_gd}, RMSE: {rmse_gd}, MAE: {mae_gd}")