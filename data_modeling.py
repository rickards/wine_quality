#%%
#Leitura do arquivo com dados
import keras
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn import preprocessing

#%%
#Visualização das 5 primeiras tuplas dos dados
df = pd.read_csv("winequality.csv", sep=";")
df.head()

#%%
#Verificação dos typos de dados
df.info()

#%%
df.describe()

# %%
# TRATAMENTO DO BANCO DE DADOS
#type: object (vamos mudar para inteiro)
#alcohol: object (vamos mudar para float e eliminar dados incosistentes)
df['type'] = df['type'].map( {'White': 1, 'Red': 0} ).astype(int)
#vamos corrigir os erros da coluna, e substituí-los para NaN, em seguida preencher essas lacunas com a moda
#poderiamos ignorá-los, mas cada dado é valioso e vamos investigar usando essas dados corrompidos
df['alcohol'] = pd.to_numeric(df['alcohol'], errors='coerce')
freq_port = df.alcohol.dropna().mode()[0]
df['alcohol'] = df['alcohol'].fillna(freq_port)

#%%
# REMOVER OUTLIERS
z_scores = stats.zscore(df) #calculate z-scores of `df` média 0 desvio padrão 1
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]

#%%
# BOXPLOTS
boxplot = df.boxplot(column=list(df.columns[:-1]))


#%%
# NORMALIZAÇÃO
# Vamos normalizar os dados para melhorar a modelagem e processamento dos dados
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_normalized = pd.DataFrame(x_scaled, columns=df.columns)
df_normalized['quality'] = df['quality']
df = df_normalized

#%%
# Passo importante vamos avaliar cada atributo
df[['quality', 'type']].groupby(['quality'], as_index=False).mean().sort_values(by='quality', ascending=False)
# Atributo type parece mostrar que a incidência de vinhos de qualidade maior estão mais presentes no vinho branco.

#%%
# Para interpretar o próximo atributo é preferível usar outra abordagem
ax = sns.violinplot(x="quality", y="fixed acidity", data=df)
# Esse atributo não parece descritivo para o modelo do vinho
# Vamos removelo do df
df = df.drop(columns=['fixed acidity'])

#%%
ax = sns.boxplot(x="quality", y="volatile acidity", data=df)
# Esse atributo já mostrou uma variedada correlacionada a qualidade, apesar de pouca
# Interessante atributo, correlação decrescente

#%%
ax = sns.boxplot(x="quality", y="citric acid", data=df)
df = df.drop(columns=['citric acid'])
# Vamos eliminar alguns atributos e deixar apenas os mais relevantes

#%%
ax = sns.boxplot(x="quality", y="residual sugar", data=df)
df = df.drop(columns=['residual sugar'])

#%%
ax = sns.boxplot(x="quality", y="chlorides", data=df)
df = df.drop(columns=['chlorides'])
# visualmente não interessante

#%%
ax = sns.boxplot(x="quality", y="free sulfur dioxide", data=df)
# correlação decrescente

#%%
ax = sns.boxplot(x="quality", y="total sulfur dioxide", data=df)
# algumas características só aparecem em vinhos com qualidade inferior

#%%
ax = sns.violinplot(x="quality", y="density", data=df)

#%%
# vamos categorizar, agrupar alguns intervalos
df.loc[ df['density'] <= 0.2, 'density'] = 0
df.loc[ df['density'] > 0.2, 'density'] = 1
df['density'] = df['density'].astype(int)
ax = sns.boxplot(x="quality", y="density", data=df)

#%%
ax = sns.violinplot(x="quality", y="pH", data=df)

#%%
# vamos categorizar, agrupar alguns intervalos
df.loc[ df['pH'] > 0.8, 'pH'] = 4
df.loc[(df['pH'] > 0.6) & (df['pH'] <= 0.8), 'pH'] = 3
df.loc[(df['pH'] > 0.4) & (df['pH'] <= 0.6), 'pH'] = 2
df.loc[(df['pH'] > 0.3) & (df['pH'] <= 0.4), 'pH'] = 1
df.loc[ df['pH'] <= 0.3, 'pH'] = 0
df['pH'] = df['pH'].astype(int)
ax = sns.boxplot(x="quality", y="pH", data=df)

#%%
ax = sns.boxplot(x="quality", y="sulphates", data=df)
df = df.drop(columns=['sulphates'])
# Não parece ser um bom atributo

#%%
ax = sns.boxplot(x="quality", y="alcohol", data=df)
# Ótimo atributo, mostra-se bom descriminante para bons vinhos


# %%
df.to_csv("winequality_cleaned.csv", sep=";", index=False)

def import_data_wine():
    return pd.read_csv("winequality_cleaned.csv", sep=";")