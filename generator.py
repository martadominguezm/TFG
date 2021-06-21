import torch
from torch import nn, optim # nn = neuronal network
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import real_data_target

def noise(quantity, size):
    return Variable(torch.randn(quantity, size)) #ruido aleatorio que el generador va a tener que convertir


#creamos la clase GeneratorNet que hereda de torch.nn.Module
class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    #creamos la arquitectura
    #__init__ inicializa una instancia de una clase o un objeto
    # atributos: out_features, leakyRelu, hidden_layers, in_features, escalonate)
    def __init__(self, out_features, leakyRelu=0.2, hidden_layers=[256, 512, 1024], in_features=100, escalonate=False):
        super(GeneratorNet, self).__init__() #para construir el modelo
        
        #le asignamos valor a algunos atributos de la clase GeneratorNet
        #los atributos definen estados
        self.in_features = in_features #en self.in_features, in_features es un atributo del objeto en la variable self
        self.layers = hidden_layers.copy() #le asisgna a las capas una copia de las ocultas (lista)
        #le inserta un rango de tamañosa las capas??? desde 0 a lo que ocupe cada muestra de entrada
        self.layers.insert(0, self.in_features)
      

      
        #igual que el del discriminador pero sin pasarle al final la capa dropout
        for count in range(0, len(self.layers)-1):
            self.add_module("hidden_" + str(count), 
                nn.Sequential(
                    nn.Linear(self.layers[count], self.layers[count+1]),
                    nn.LeakyReLU(leakyRelu)
                )
            )

        if not escalonate: #si no está escalonado
            self.add_module("out", 
                nn.Sequential(
                    nn.Linear(self.layers[-1], out_features)
                )
            )
        else:
            self.add_module("out", 
                nn.Sequential(
                    nn.Linear(self.layers[-1], out_features),
                    escalonate
                )
            )
    
    def forward(self, x): #describe como se ha calculado el output del modelo
        for name, module in self.named_children():
            x = module(x)
        return x

    def create_data(self, quantity):#debe crear una tabla llamada quantity?
        points = noise(quantity, self.in_features)
        try:
            data=self.forward(points.cuda()) #se ejecuta en gpu??
        except RuntimeError:
            data=self.forward(points.cpu()) #se ejecuta en cpu??
        return data.detach().numpy() #obtenemos matriz numérica a partir de un tensor torch


#ENTRENAMIENTO GAN
def train_generator(optimizer, discriminator, loss, fake_data):

    # 2. Train Generator
    # Reset gradients (ponemos gradientes a 0)
    optimizer.zero_grad()

    # Sample noise and generate fake data
    prediction = discriminator(fake_data)

    # Calculate error and backpropagate

    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward() #retropropagación

    # Update weights with gradients
    optimizer.step()
    
    # Return error
    return error