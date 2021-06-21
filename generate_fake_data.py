import torch
import pandas as pd
import matplotlib.pyplot as plt
import glob
import ipywidgets as widgets
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from data_treatment import DataSet, DataAtts
from discriminator import *
from generator import *
from IPython.display import display
import glob

def generate_fake_data(original_db, original_db_name, folder_name):
  for file in glob.glob(folder_name):
    name = file.split("/")[-1][10:-3]
    try:
        checkpoint= torch.load(file, map_location='cuda')
    except:
        checkpoint= torch.load(file, map_location='cpu')
    generator = GeneratorNet(**checkpoint['model_attributes'])
    generator.load_state_dict(checkpoint['model_state_dict'])
    size = original_db.shape[0]
    new_data = generator.create_data(size)
    df = pd.DataFrame(new_data, columns=original_db.columns)
    name = name + "_size-" + str(size)
    df.to_csv( "fake_data/" + original_db_name + "/" + name + ".csv", index=False)
    print("Fake data: " + name)
