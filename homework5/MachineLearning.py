# -*- coding: utf-8 -*-
#clean this up
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.neural_network import MLPClassifier
import csv


"""
Methods to test
- Logisitic Regression
- Neural Network
- Kernel SVM
"""

def main():
    #load data
    X, y = load_csv_data('all/train.csv')

    #preprocess and split dataset
    X = preprocess(X)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    #define models and set hyperparams
    SEED = 7007
    models = {

        'Logistic Regression' : LogisticRegression(max_iter=500, verbose=0, multi_class="multinomial"),
        'SVM' : SVC( kernel='poly', verbose=0), #can add more hyperparams here
        'Random Forest': RandomForestClassifier(n_estimators=500,
                                           n_jobs=-1,
                                            random_state=SEED),
        'Extra Tree': ExtraTreesClassifier(
           max_depth=400,
           n_estimators=500, n_jobs=-1,
           oob_score=False, random_state=SEED,
           warm_start=True),

        'Neural Network' : MLPClassifier(solver='adam', max_iter=500, learning_rate_init=0.01, alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)

    }

    #run cross validation to decide the best model to train
    print("\n")
    results = cross_val(models, X, y)
    best_model_name = choose_best(results)
    model = models[best_model_name]

    print("\nThe best model is: ", best_model_name)


    #train model
    model.fit(X, y)

    #make prediction and insert them into csv
    y_pred = model.predict(X)
    write_csv(y_pred)
    print("Written to csv!\n")
    #print("Score: %.2f" % lr.score(X_test, y_test))
    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
    return


def cross_val(models, X_train, y_train):
    r = dict()
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=4, scoring='accuracy')
        r[name] = scores
        print(name, 'Accuracy Mean {0:.4f}, Std {1:.4f}'.format(
              scores.mean(), scores.std()))
    return r

def choose_best(results):
    r = dict()
    for name, arr in results.items():
        r[name] = arr.mean()

    best_model =  [m for m, e in r.items() if e == max(r.values())][0]
    return best_model




#preprocess, reduce data size and normalise
#
def preprocess(data):
    X_r = reduce_dim(data,10,14,14,-1)
    X_r = np.asarray(data)
    X = normalize(X_r, norm='l2')
    return X

#reduce_dim to fold the columns which are redundant
def reduce_dim(data,i,j,p,q):
    rdata=[]
    for row in data:
        x=np.argmax(row[i:j])+1
        y=np.argmax(row[p:q])+1
        new_row = list(row[:i]) + [x,y] + list(row[q:])
        #new_row = row[:i].reshape(len(row[:i]), 1)+[x,y]+row[q:].reshape(len(row[q:]), 1)
        rdata.append(new_row)

    return(rdata)

# Load Features
def load_csv_test(filename):
    file = pd.read_csv(filename)

    data = file[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
                 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1',
                 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
                 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
                 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
                 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
                 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
                 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
                 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
                 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']]
    data = np.array(data)
    return data

#Load Labels and features
def load_csv_data(filename):
    file = pd.read_csv(filename)

    data = file[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
                 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1',
                 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
                 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
                 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
                 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
                 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
                 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
                 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
                 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']]
    labels = file['Cover_Type']
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# write csv
def write_csv(predict):
    with open("all/predict.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Id", "Cover_Type"])
        for i in range(0, len(predict)):
            writer.writerow([(15121 + i), predict[i]])

if __name__ == '__main__':
    main()
