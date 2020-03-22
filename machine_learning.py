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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from data_modeling import import_data_wine

df = import_data_wine()

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
    interval = int(len(df)/10)
    total_acc = []
    total_rmse = []
    for i in range(0, len(df)-interval, interval):

        test_df = df.loc[i:i+interval]
        if i+interval > len(df)-interval:
            test_df = df.loc[i:]
        train_df = df.drop(test_df.index)

        X_test = test_df.drop("quality", axis=1)
        Y_test = test_df["quality"]

        X_train = train_df.drop("quality", axis=1)
        Y_train = train_df["quality"]

        # treina
        tech.fit(X_train, Y_train)
        
        # predição e acc
        # print(X_test)
        Y_pred = tech.predict(X_test)
        print(Y_pred)
        
        acc_tech = round(tech.score(X_test, Y_test) * 100, 2)
        mse = mean_squared_error(Y_test, Y_pred)
        # print(f'acc: {acc_tech} mse:{mse}')
        total_acc.append(acc_tech)
        total_rmse.append(mse)

    acc_mean = sum(total_acc)/len(total_acc)
    rmse_mean = sum(total_rmse)/len(total_rmse)
    print(f'acc: {round(acc_mean, 2)} - rmse: {round(rmse_mean, 2)} {tech.__class__.__name__}')
    # input()
