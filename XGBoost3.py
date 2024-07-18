import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import TargetEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from yellowbrick.classifier import ConfusionMatrix


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


#Target Encoder 

encoder = TargetEncoder()
X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)


estimators = [
    ('clf', XGBClassifier(random_state = 8))
]

pipe = Pipeline(steps=estimators)


search_space = {
    'clf__max_depth': Integer(2,8),
    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode' : Real(0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0)
}

opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=100, random_state=8) 
# in reality, you may consider setting cv and n_iter to higher values

opt_result = opt.fit(X_train, y_train)

best_model = opt_result.best_estimator_

y_pred = best_model.predict(X_test)

y_pred = [np.argmax(v) for v in y_pred]

arquivo = open("saida.txt", "w+")
print(classification_report(y_test,y_pred), file = arquivo)
print(f"Best parameters found: {opt_result.best_params_}", file=arquivo)
print(f"Best cross-validation accuracy: {opt_result.best_score_}", file=arquivo)



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
iris_cm.show(outpath="XBoost_3.png")

