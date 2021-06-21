import numpy as np
import pandas as pd
from data_treatment import DataAtts
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import export_graphviz # Decision tree from sklearn

import pydotplus # Decision tree plotting
from IPython.display import Image

import ipywidgets as widgets
import glob


def prepare_train_and_test_dataset(dataAtts,data,fake_data):
  #data(filas,columnas)==data(0,1), es decir,
  #data.shape[0] devuelve el número de filas del dataset
  
  #Para el dataset _escalonated con todos los datos 
  original_data_training_set = data.head(int(data.shape[0]*0.7))
  fake_data_training_set  = fake_data.head(int(fake_data.shape[0]*0.7))
  original_data_testing_set  = data.tail(int(data.shape[0]*0.3))
  fake_data_testing_set  = fake_data.tail(int(fake_data.shape[0]*0.3))

  #Para el dataset de prueba reducido a la mitad
  #original_data_training_set = pd.concat([data.head(int(data.shape[0]*0.46)),data.tail(int(data.shape[0]*0.24))])
  #fake_data_training_set  = pd.concat([fake_data.head(int(fake_data.shape[0]*0.46)),fake_data.tail(int(fake_data.shape[0]*0.24))])
  #original_data_testing_set  = pd.concat([data.head(int(data.shape[0]*0.2)),data.tail(int(data.shape[0]*0.1))])
  #fake_data_testing_set  = pd.concat([fake_data.head(int(fake_data.shape[0]*0.2)),fake_data.tail(int(fake_data.shape[0]*0.1))])


  mixed_data_training_set=pd.concat([original_data_training_set, fake_data_training_set])
  mixed_data_testing_set=pd.concat([original_data_testing_set, fake_data_testing_set])

  mask_0 = original_data_testing_set[dataAtts.class_name] == 0
  mask_1 = original_data_testing_set[dataAtts.class_name] == 1
  original_1s = original_data_testing_set[mask_1]
  head_0s = original_data_testing_set[mask_0].head(original_1s.shape[0])
  tail_0s = original_data_testing_set[mask_0].tail(original_1s.shape[0])


  sampled_0s = original_data_testing_set[mask_0].sample(original_1s.shape[0])
  balanced_test = pd.concat([original_1s, sampled_0s])

  #train = original_data_training_set
  train = fake_data_training_set
  test = original_data_testing_set

  trainX = train.drop(dataAtts.class_name, 1)
  testX = test.drop(dataAtts.class_name, 1)
  y_train = train[dataAtts.class_name]
  y_test = test[dataAtts.class_name]
  
  return trainX,testX,y_train,y_test


def plot_classifier(trainX,testX,y_train,y_test,depth=3, min_leaves=1):
  clf1 = DT(max_depth = depth, min_samples_leaf=min_leaves)
  clf1 = clf1.fit(trainX, y_train)

  export_graphviz(clf1, out_file="models/tree.dot", feature_names=trainX.columns, class_names=["0","1"], filled=True, rounded=True)
  g = pydotplus.graph_from_dot_file(path="models/tree.dot")
  Image(g.create_png()) #le he puesto return que sino no iba
  
  pred = clf1.predict_proba(testX)
  #if pred.shape[1] > 1:
  pred = np.argmax(pred, axis=1)
  #else:
      #pred = pred.reshape((pred.shape[0]))
      #if negative=="0":
        #pred = pred-1
  mse = ((pred - y_test.values)**2).mean(axis=0)
  print("Prediction error: ", mse)

  return pred, g

def conf_m(conf_matrix):
  TN, FN, TP, FP = conf_matrix[0][0], conf_matrix[1][0], conf_matrix[1][1], conf_matrix[0][1]
  confusion_matrix_str = str(TN) + "/" + str(FN) + "/" + str(TP) + "/" + str(FP)
  precision = round(TP/(TP+FP), 3)
  recall = round(TP/(TP+FN), 3)
  accuracy = round((TP+TN)/(TP+TN+FP+FN), 3)
  f1_score = round(2*(precision*recall)/(precision+recall),3)
  print("TN/FN/TP/FP: ", confusion_matrix_str)
  print("Accuracy: ", accuracy)
  print("Precision: ", precision)
  print("Recall: ", recall)
  print("F-1 score: ", f1_score)
