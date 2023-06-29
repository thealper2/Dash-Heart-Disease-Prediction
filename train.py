import numpy as np
import pandas as pd
df = pd.read_csv("data/heart_cleveland_upload.csv")

X = df.drop("condition", axis=1)
y = df["condition"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier()
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
accuracy_score(y_test, y_pred)

import pickle
pickle.dump(et, open("models/extratrees.pkl", "wb"))
