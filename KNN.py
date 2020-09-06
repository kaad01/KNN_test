import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("Car Data Set/car.data")

le = preprocessing.LabelEncoder()                           #create LabelEncoder()
#Transform the attributes into ints with different classes
buying = le.fit_transform(data["buying"])
maint = le.fit_transform(data["maint"])
door = le.fit_transform(data["door"])
persons = le.fit_transform(data["persons"])
lug_boot = le.fit_transform(data["lug_boot"])
safety = le.fit_transform(data["safety"])
cls = le.fit_transform(data["class"])


predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))   #One big list of all the attributes
y = list(cls)

#Splitting the data into sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#Creating the model and doing the math
model = KNeighborsClassifier(n_neighbors = 9)
model.fit(x_train, y_train)                                     #training
acc = model.score(x_test, y_test)                               #testing
print(acc)

#showing the test
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(x_test)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    #showing the nearest Neighbor
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)
