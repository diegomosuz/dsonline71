import numpy as np
import pickle

local_scaler = pickle.load(open('scaler.pickle', 'rb'))

local_classifier = pickle.load(open('classifier.pickle', 'rb'))


# Sin necesidad de entrenar 

new_pred = classifier.predict(sc.transform(np.array([[40, 20000]])))
new_pred_prob = classifier.predict_proba(sc.transform(np.array([[40, 20000]])))[:,1]