import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import seaborn as sns
import matplotlib as plt
from scipy.stats import pointbiserialr, spearmanr
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier





data = pd.read_csv("./income2022f/train_final.csv")
data.head()





data.describe()





data.isnull().sum()





sns.heatmap(data.corr(), annot = True)





data["workclass"] = data["workclass"].replace("?", "Private")
data["occupation"] = data["occupation"].replace("?", "Prof-speciality")
data["native.country"] = data["native.country"].replace("?", "United-States")





data.skew()





from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()





data.columns
data["workclass"] = le.fit_transform(data["workclass"])
data["education"] = le.fit_transform(data["education"])
data["marital.status"] = le.fit_transform(data["marital.status"])
data["occupation"] = le.fit_transform(data["occupation"])
data["relationship"] = le.fit_transform(data["relationship"])
data["native.country"] = le.fit_transform(data["native.country"])
data["race"] = le.fit_transform(data["race"])
data["sex"] = le.fit_transform(data["sex"])
data





from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

ds_x = data.drop("income>50K", axis = "columns")
y = data["income>50K"]
dataset = sc.fit_transform(ds_x)
x = pd.DataFrame(dataset, columns = ds_x.columns)
x





from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix





lg = LogisticRegression()
svc = SVC()
knn = KNeighborsClassifier()





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 11)

# lg = LogisticRegression()
# lg.fit(x_train, y_train)


# y_pred_test = lg.predict(x_test)
# lg_acc = accuracy_score(y_test, y_pred_test)
# print("Accuracy of LR is:",lg_acc)
# print(classification_report(y_test, y_pred_test))

# svc = SVC()
# svc.fit(x_train, y_train)

# y_pred_test = svc.predict(x_test)
# svc_acc = accuracy_score(y_test, y_pred_test)
# print("Accuracy of SVC is:",svc_acc)
# print(classification_report(y_test, y_pred_test))


# knn = KNeighborsClassifier()
# knn.fit(x_train, y_train)

# y_pred_test = knn.predict(x_test)
# knn_acc = accuracy_score(y_test, y_pred_test)
# print("Accuracy of knn is:",knn_acc)
# print(classification_report(y_test, y_pred_test))


# for i in [rfc, ad, gd]:
#     i.fit(x_train, y_train)
#     pred = i.predict(x_test)
#     test_score = accuracy_score(y_test, pred)
#     train_score = accuracy_score(y_train, i.predict(x_train))
#     print(rfc, accuracy_score(y_test, pred))
#     print(rfc, accuracy_score(y_train, i.predict(x_train)))
#     if abs(train_score - test_score) <= 0.01:
#         print(i)
#         print("accuracy score for train data", accuracy_score(y_test, pred))
#         print("accuracy score for test data", accuracy_score(y_train, i.predict(x_train)))
#         print(classification_report(y_test, pred))
#         print(confusion_matrix(y_test, pred))







from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier



rfc = RandomForestClassifier()
ad = AdaBoostClassifier()
gd = GradientBoostingClassifier()


# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 11)


# rfc = RandomForestClassifier()
# rfc.fit(x_train, y_train)


# y_pred_test = rfc.predict(x_test)
# rfc_acc = accuracy_score(y_test, y_pred_test)
# print("Accuracy of rfc is:", rfc_acc)
# print(classification_report(y_test, y_pred_test))

# ad = AdaBoostClassifier()
# ad.fit(x_train, y_train)

# y_pred_test = ad.predict(x_test)
# ad_acc = accuracy_score(y_test, y_pred_test)
# print("Accuracy of ad is:", ad_acc)
# print(classification_report(y_test, y_pred_test))


# gd = GradientBoostingClassifier()
# gd.fit(x_train, y_train)

# y_pred_test = gd.predict(x_test)
# gd_acc = accuracy_score(y_test, y_pred_test)
# print("Accuracy of gd is:", gd_acc)
# print(classification_report(y_test, y_pred_test))






for i in [rfc, ad, gd]:
    i.fit(x_train, y_train)
    pred = i.predict(x_test)
    test_score = accuracy_score(y_test, pred)
    train_score = accuracy_score(y_train, i.predict(x_train))
    print(rfc, "accuracy score for train data",accuracy_score(y_test, pred))
    print(rfc,"accuracy score for train data", accuracy_score(y_train, i.predict(x_train)))
    if abs(train_score - test_score) <= 0.01:
        print(i)
        print("accuracy score for train data", accuracy_score(y_test, pred))
        print("accuracy score for test data", accuracy_score(y_train, i.predict(x_train)))
        print(classification_report(y_test, pred))
        print(confusion_matrix(y_test, pred))





from sklearn.model_selection import cross_val_score


for i in range(2,9):
    cv = cross_val_score(gd, x, y, cv = i)
    print(gd, cv.mean())
