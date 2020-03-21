#%%
#Leitura do arquivo com dados
import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

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
# df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

#%%
# BOXPLOTS
boxplot = df.boxplot(column=list(df.columns[:-1]))


#%%
# NORMALIZAÇÃO
# Vamos normalizar os dados para melhorar a modelagem e processamento dos dados
from sklearn import preprocessing

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

#%%
ax = sns.boxplot(x="quality", y="free sulfur dioxide", data=df)
# correlação decrescente

#%%
ax = sns.boxplot(x="quality", y="total sulfur dioxide", data=df)
df = df.drop(columns=['total sulfur dioxide'])
# eliminando atributos fracos

#%%
ax = sns.violinplot(x="quality", y="density", data=df)

#%%
ax = sns.violinplot(x="quality", y="pH", data=df)

#%%
# vamos categorizar, agrupar alguns intervalos
df.loc[ df['pH'] <= 0.3, 'pH'] = 0
df.loc[(df['pH'] > 0.3) & (df['pH'] <= 0.4), 'pH'] = 1
df.loc[(df['pH'] > 0.4) & (df['pH'] <= 0.6), 'pH'] = 2
df.loc[(df['pH'] > 0.6) & (df['pH'] <= 0.8), 'pH'] = 3
df.loc[ df['pH'] > 0.8, 'pH'] = 4
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
machine_learning_techniques = []
machine_learning_techniques.append(LogisticRegression())
machine_learning_techniques.append(SVC())
machine_learning_techniques.append(LinearSVC())
machine_learning_techniques.append(RandomForestClassifier())
machine_learning_techniques.append(KNeighborsClassifier())
machine_learning_techniques.append(GaussianNB())
machine_learning_techniques.append(Perceptron())
machine_learning_techniques.append(SGDClassifier())
machine_learning_techniques.append(DecisionTreeClassifier())
machine_learning_techniques.append(MultinomialNB())

df = df.sample(frac=1).reset_index(drop=True)
for tech in machine_learning_techniques:

    # cross 10-fold validation
    interval = int(len(df)/9)
    total_acc = []
    for i in range(0, len(df)+len(df)%9, interval):
        test_df = df.loc[i:i+interval]
        train_df = df.drop(test_df.index)

        X_test = test_df.drop("quality", axis=1)
        Y_test = test_df["quality"]

        X_train = train_df.drop("quality", axis=1)
        Y_train = train_df["quality"]

        # treina
        tech.fit(X_train, Y_train)
        
        # predição e acc
        Y_pred = tech.predict(X_test)
        acc_tech = round(tech.score(X_test, Y_test) * 100, 2)
        total_acc.append(acc_tech)

    print(f'{sum(total_acc)/len(total_acc)} {tech.__class__.__name__}')


# %%
# DEEP LEARNING
import keras

from keras.utils import np_utils

model = keras.Sequential([
    keras.layers.Dense(64, activation='elu', input_shape=(7,)),
    keras.layers.GaussianDropout(0.2),
    keras.layers.Dense(64, activation='tanh'),
    # keras.layers.Embedding(1000, 64, input_length=7),
    # keras.layers.GRU(128, activation='elu'),
    keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
df = df.sample(frac=1).reset_index(drop=True)
interval = int(len(df)/10)
print(f'interval: {interval}')

test_df = df.loc[:interval]
train_df = df.drop(test_df.index)

X_test = test_df.drop("quality", axis=1)
Y_test = test_df["quality"]-3

X_train = train_df.drop("quality", axis=1)
Y_train = train_df["quality"]-3

print(np_utils.to_categorical(Y_test).shape)

# treina
model.fit(X_train, np_utils.to_categorical(Y_train, num_classes=7), epochs=100, verbose=0)

# predição e acc
_, acc_tech = model.evaluate(X_test, np_utils.to_categorical(Y_test, num_classes=7), verbose=1)
print(acc_tech)

# %%
