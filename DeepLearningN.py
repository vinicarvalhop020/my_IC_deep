import numpy as np
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report
import pandas as pd
from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from skopt import BayesSearchCV



Data = pd.read_csv(r'C:\Users\Vini\Documents\IC\DeepLearning1st\inca15.csv',low_memory = False)
Data = Data.drop(Data[Data['estadofinal'] > 8].index)
Data.loc[Data['idade1'] < 0,'idade1'] = int(Data['idade1'][Data['idade1'] > 0].mean())
Data = Data.drop(Data[Data['racacor'] >= 99].index)


# Definir um dicionário para mapear os valores a serem substituídos
mapeamento = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 8: 5}
Data['estadofinal'] = Data['estadofinal'].replace(mapeamento)

X = Data.drop(columns='estadofinal')
y = Data['estadofinal']

X_train, X_test, y_train , y_test  = train_test_split(X, y, test_size = 0.20, stratify=y ,random_state = 8)


#Target Encoder Aumenta o numero de colunas 

encoder = TargetEncoder()
X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)


def create_model(optimizer='adam', units=128, units1=64, units2=32, units3 = 16, dropout_rate=0.2, activation ='relu'):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)), 
        tf.keras.layers.Dense(units, activation = activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(units1,  activation = activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(units2, activation= activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(units3, activation= activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def adaptive_learning_rate(epoch, lr):
    if epoch < 80:
        return lr
    else:
        return lr * np.exp(-0.1)
 

model = KerasClassifier(model=create_model, verbose=1)

param_grid = {
    'epochs': [100],
    'batch_size': [16],
    'model__activation':['relu','sigmoid', 'tanh']  
}

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(adaptive_learning_rate)

grid = BayesSearchCV(model, param_grid, cv=3, n_iter=100, random_state=8) 

grid_result = grid.fit(X_train, y_train, callbacks = [lr_scheduler])

best_model = grid_result.best_estimator_

y_pred = best_model.predict(X_test)

with open("DeepLearningParamGRID.txt", "w+") as arquivo:
    print(classification_report(y_test, y_pred), file=arquivo)
    print(f"Best parameters found: {grid_result.best_params_}", file=arquivo)
    print(f"Best cross-validation accuracy: {grid_result.best_score_}", file=arquivo)


iris_cm = ConfusionMatrix(
    best_model, classes = [0,1,2,3,4,5],
    label_encoder={0: "0: Sem evidência da doença",
    1: "1: Remissão parcial",
    2: "2 : Doença estável",
    3: "3: Doença em progressão",
    4: "4: Suporte terapêutico oncológico",
    5: "5: Não se aplica"}
)

iris_cm.fit(X_train, y_train)
iris_cm.score(X_test, y_test)
iris_cm.show(outpath="DeepLN.png")

