import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv("storepurchasedata.csv")

data.describe()

X = data.iloc[:, :-1].values

y = data.iloc[:,-1].values


# Separamos el dataset en dos porciones, una para entrenamiento y otra para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Escalamos los datos para que el modelo no se segue con una de las variables
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creamos el clasificador basado en aprendizaje supervisado
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)[:,1]

# Evaluamos el modelo en tiempo de desarrollo

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)


report = classification_report(y_test, y_pred)


# ----- VER CUAL ES EL PROCEDIMIENTO SUPONIENDO QUE EL MODELO ESTE PROD

new_pred = classifier.predict(sc.transform(np.array([[40, 200000]])))
new_pred_prob = classifier.predict_proba(sc.transform(np.array([[40, 200000]])))[:,1]

# Preliminares para llevar este modelo a producción

model_file = "classifier.pickle"
pickle.dump(classifier, open(model_file, 'wb'))

scaler_file = "scaler.pickle"
pickle.dump(sc, open(scaler_file, 'wb'))













































