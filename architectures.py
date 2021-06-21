import torch
import pandas as pd
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from data_treatment import DataSet, DataAtts
from generator import *
from discriminator import *

import os  
import glob
from utils import * 

class Architecture():
    def __init__(self, learning_rate, batch_size, loss, hidden_layers, name):
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.loss=loss
        self.hidden_layers=hidden_layers
        self.name=name

def save_model(name, epoch, attributes, dictionary, optimizer_dictionary, loss_function, db_name, arch_name):
    torch.save({
        'epoch': epoch,
        'model_attributes': attributes,
        'model_state_dict': dictionary,
        'optimizer_state_dict': optimizer_dictionary,
        'loss': loss_function
    }, "models/" + db_name + "/" + name + "_" + arch_name + ".pt")



def set_architectures(learning_rate,batch_size,hidden_layers,number_of_experiments,num_epochs):
  #Creación de las diferentes arquitecturas DL
  architectures=[]
  count=0
  for lr in learning_rate:
      for b_size in batch_size:
          for hidden in hidden_layers:
              for i in range(number_of_experiments):
                  name = "id-" + str(count)
                  name += "_epochs-" + str(num_epochs[0])
                  name += "_layer-" + str(len(hidden))
                  name += "_lr-" + str(lr)
                  name += "_batch-" + str(b_size)
                  name += "_arc-" + ','.join(map(str, hidden))
                  architectures.append( Architecture(
                          learning_rate=lr,
                          batch_size=b_size,
                          loss=nn.BCELoss(),
                          hidden_layers=hidden,
                          name=name
                      )
                  )
                  count+=1
  return architectures