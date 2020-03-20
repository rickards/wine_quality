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
#vamos corrigir os erros da coluna, e substituí-los para NaN, em seguida preencher essas lacunas com a moda
#poderiamos ignorá-los, mas cada dado é valioso e vamos investigar usando essas dados corrompidos
df['alcohol'] = pd.to_numeric(df['alcohol'], errors='coerce')
freq_port = df.alcohol.dropna().mode()[0]
df['alcohol'] = df['alcohol'].fillna(freq_port)

#%%
#Vamos normalizar os dados para melhorar a visualização dos boxplots
from sklearn import preprocessing

x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_normalized = pd.DataFrame(x_scaled, columns=df.columns)
df_normalized['quality'] = df['quality']
df = df_normalized

#%%
boxplot = df.boxplot(column=list(df.columns[:-1]))


# %%
# vaoms usar árvore de decisão para avaliar acc desse modelo
from sklearn.tree import DecisionTreeClassifier

# cross 10-fold validation
interval = int(len(df)/10)
for i in range(0, len(df)+len(df)%10, interval):
    test_df = df.loc[i:i+interval]
    train_df = df.drop(test_df.index)

    X_test = test_df.drop("quality", axis=1)
    Y_test = test_df["quality"]

    X_train = train_df.drop("quality", axis=1)
    Y_train = train_df["quality"]

    # treina
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    
    # predição e acc
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
    print(acc_decision_tree)