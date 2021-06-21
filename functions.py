import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time 
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from scipy.stats import norm
from data_treatment import DataSet, DataAtts
from generator import *
from discriminator import *

def data_analysis(dataAtts, file_name):
  data = pd.read_csv(file_name)
  print(dataAtts.message, "\n")
  print("Original Data")
  print(dataAtts.values_names[0], round(data[dataAtts.class_name].value_counts()[0]/len(data) * 100,2), '%  of the dataset')
  print(dataAtts.values_names[1], round(data[dataAtts.class_name].value_counts()[1]/len(data) * 100,2), '%  of the dataset')

def getTime():
  return time.time()

def totalTime(initial_time):
  print("time:", time.time()-initial_time)

def compare_errors(d_error_plt,g_error_plt,diff_error_plt,dataAtts,arc):
  #Display plots
  file = open("results/" + dataAtts.fname + "/" + arc.name +"_"+"error_growth.txt", "w")
  fig = plt.figure(figsize=(20, 2))
  ax = fig.add_subplot(111)
  ax.plot(d_error_plt, 'b')
  ax.plot(g_error_plt, 'r')
  ax.plot(diff_error_plt, 'g')

  file.write("Discriminator error: " + str(d_error_plt) + "\n")
  file.write("\n\n\n")
  file.write("Generator error: " + str(g_error_plt) + "\n")
  file.write("\n\n\n")
  file.write("Diff error: " + str(diff_error_plt) + "\n")
  file.close()


  plt.savefig('images/'+ dataAtts.fname + "/"+ arc.name +"_"+'error.png')
  plt.show() #muestra las gráficas
  plt.clf()

def test_analysis(dataAtts,data):
  test = data.tail(int(data.shape[0]*0.3))
  print("\nTest Data")
  print("Outcome = 0: ", round(test[dataAtts.class_name].value_counts()[0]/len(test) * 100,2), '%  of the dataset')
  print("Outcome = 1: ", round(test[dataAtts.class_name].value_counts()[1]/len(test) * 100,2), '%  of the dataset')


def fake_data_analysis(dataAtts,fake_files_dropdown):
  fake_data = pd.read_csv(fake_files_dropdown.value)
  fake_data.loc[getattr(fake_data, dataAtts.class_name) >= 0.5, dataAtts.class_name] = 1
  fake_data.loc[getattr(fake_data, dataAtts.class_name) < 0.5, dataAtts.class_name] = 0

  print("\nFake Data")
  try:
      positive=str(round(fake_data[dataAtts.class_name].value_counts()[0]/len(fake_data) * 100,2))
  except:
      positive="0"
  try:
      negative=str(round(fake_data[dataAtts.class_name].value_counts()[1]/len(fake_data) * 100,2))
  except:
      negative="0"
      

  print("Outcome = 0: ", positive, '%  of the dataset')
  print("Outcome = 1: ", negative, '%  of the dataset')


def compare_distribution_fake_data(dataAtts,file_name,fake_files_dropdown):
  data = pd.read_csv(file_name)
  classes = list(data)
  fake_data = pd.read_csv(fake_files_dropdown.value)
  fake_data.loc[getattr(fake_data, dataAtts.class_name) >= 0.5, dataAtts.class_name] = 1
  fake_data.loc[getattr(fake_data, dataAtts.class_name) < 0.5, dataAtts.class_name] = 0

  for name in classes:
      if name=="Unnamed: 32":
          continue
          
      plt.xlabel('Values')
      plt.ylabel('Probability')
      plt.title(name + " distribution")
      real_dist = data[name].values
      fake_dist = fake_data[name].values
      plt.hist(real_dist, 50, density=True, alpha=0.5, facecolor='r')
      plt.hist(fake_dist, 50, density=True, alpha=0.5, facecolor='g')
      plt.savefig('fake_data/'+ dataAtts.fname + "/"+name+'_distribution.png')
      plt.show()
      plt.clf()
