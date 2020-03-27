# %%
# DEEP LEARNING
import keras
import keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils
from data_modeling import import_data_wine, balancing
from callback_model import TerminateTrainingLoss

df = import_data_wine()

model = keras.Sequential([
    keras.layers.Dense(128, activation='elu', input_shape=(13,)),
    keras.layers.Dense(64, activation='elu'),
    keras.layers.Dense(32, activation='elu'),
    keras.layers.Dense(16, activation='elu'),
    keras.layers.Dense(7, activation='softmax')
])

def RMSE(y_true, y_pred):
    # Um pouco diferente da abordagem do sklearn por que utiliza das probabilidades da softmax 
    # (não comparável até que se utilizem a mesma abordagem para avaliação)
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

adam = keras.optimizers.Adam(learning_rate=0.004)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['acc', RMSE])

# %%
df = df.sample(frac=1).reset_index(drop=True)
interval = int(len(df)/10)
print(f'interval: {interval}')

test_df = df.loc[:interval]
train_df = df.drop(test_df.index)
# Balanceamento não melhorou a predição dos modelos e por isso não foi utilizado, seguimos com a distribuição reais dos dados.
# train_df = balancing(train_df)

# Como parte do pré processamento vamos transformar o label em matriz identidade
# por isso vamos simplificar o valor desse resultado para um intervalo melhor aceitável
# entre 0 e 7 e não entre 3 e 9
X_test = test_df.drop("quality", axis=1)
Y_test = test_df["quality"]-3

X_train = train_df.drop("quality", axis=1)
Y_train = train_df["quality"]-3

print(np_utils.to_categorical(Y_test).shape)

# callback
callback = TerminateTrainingLoss()

# treina
history = model.fit(X_train, 
                    np_utils.to_categorical(Y_train, num_classes=7), 
                    epochs=1000, 
                    callbacks=[callback],
                    verbose=1, 
                    batch_size=128, 
                    validation_split = 0.1)

# predição e acc e mse
_, acc_tech, _ = model.evaluate(X_test, 
                            np_utils.to_categorical(Y_test, num_classes=7), 
                            verbose=0)
print(f'\nDeep learning trouxe uma acurácia de {round(acc_tech,2)}\n')

# %%
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation mse values
plt.plot(history.history['RMSE'])
plt.plot(history.history['val_RMSE'])
plt.title('Model mse')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print(history.history.keys())