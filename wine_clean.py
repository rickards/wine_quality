#%%
#Leitura do arquivo com dados
import keras
import pandas as pd
%matplotlib inline

df = pd.read_csv("winequality.csv", sep=";")

#%%
#Visualização das 5 primeiras tuplas dos dados
df.head()

#%%
#Verificação dos typos de dados
df.info()

# %%
#type: object (vamos mudar para inteiro)
#alcohol: object (vamos mudar para float e eliminar dados incosistentes)
df['type'] = df['type'].map( {'White': 1, 'Red': 0} ).astype(int)
df['type'] = df['type'].map( {'White': 1, 'Red': 0} ).astype(int)

#%%
boxplot = df.boxplot(column=['alcohol'])
