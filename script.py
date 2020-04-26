def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

income_data = pd.read_csv("income.csv", header = 0, delimiter = ", ")
#print(income_data.iloc[0])
labels = income_data[["income"]]

#testing the data without the column sex
#data = income_data[["age","capital-gain", "capital-loss","hours-per-week"]]

#Pour transformer notre colonne sex en une colonne numérique binaire
income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)
#Comme les états-unis dominent cette catégorie, on va créer une colonne numérique binaire
#print(income_data["native-country"].value_counts())

income_data["country-int"]= income_data["native-country"].apply(lambda row: 1 if row == "United-States" else 0)

data = income_data[["age","capital-gain", "capital-loss","hours-per-week","sex-int"]]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state=1)

forest = RandomForestClassifier(random_state = 1)
forest.fit(train_data, train_labels)
print(forest.score(test_data, test_labels))






