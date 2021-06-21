import torch
from torch import nn, optim # nn = neuronal network
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import real_data_target, fake_data_target


#creamos la clase DiscriminatorNet que hereda de torch.nn.Module
class DiscriminatorNet(torch.nn.Module): 
    """
    A three hidden-layer discriminative neural network
    """

    #creamos la arquitectura
    # (in_features, capa de activación leakyRelu, capa dropout, capas ocultas)
    #in_features= size of each input sample (tamñano de cada muestra de entrada, no es necesario especificarlo)
    #leakyRelu== devuelve pequeño valor negativo proporcional a la entrada
    #dropout= proporción de unidades que se eliminarán de la capa anterior (regularización)
    #hidden_layers= capas ocultas por las que está formada la GAN, con 1024, 512, 256 nodos/capa
    def __init__(self, in_features, leakyRelu=0.2, dropout=0.3, hidden_layers=[1024, 512, 256]):
        super(DiscriminatorNet, self).__init__() #para construir el modelo
        
        out_features = 1 #tamaño muestra de salida inicializado a 1
        self.layers = hidden_layers.copy()  #le asisgna a las capas una copia de las ocultas
        #le inserta un rango de tamañosa las capas??? desde 0 a lo que ocupe cada muestra de entrada
        self.layers.insert(0, in_features)

        for count in range(0, len(self.layers)-1): #recorre todas las capas desde la primera hasta la última
            #crea un módulo llamado "hidden_x" para cada capa (desde la primera a la última)
            self.add_module("hidden_" + str(count), #add_module(name, module) 
                #modelo secuencial= pila lineal de capas
                nn.Sequential(   #capa Linear == (in_features,out_features)
                    #es decir crea una capa oculta entre la capa en la que está y la siguiente de forma secuencial
                    nn.Linear(self.layers[count], self.layers[count+1]),
                    #LeakyReLU soluciona los problemas cuando la salida es 0 (gradiente siempre distinto de 0)
                    nn.LeakyReLU(leakyRelu), #LeakyReLU devuelve un pequeño número negativo proporcional a la entrada
                    #para evitar sobreajuste->técnicas de regularización-> uso capas dropout ("capas de abandono")
                    #sobreajuste == algoritmo funciona bien en el conjunto de datos de entrenamiento, pero no en el conjunto de datos de prueba
                    nn.Dropout(dropout)#elige un conjunto aleatorio de unidades de la capa anterior y establece su salida en cero
                    #red  entrenada para producir predicciones precisas incluso en condiciones desconocidas
                )
            )
        
        #crea un módulo "out"
        self.add_module("out", #add_module(name, module)
            #modelo secuencial= pila lineal de capas
            nn.Sequential( #capa Linear == (in_features,out_features)
                nn.Linear(self.layers[-1], out_features), # ??
                #La activación sigmoidea es útil si desea escalar la salida de la capa entre 0 y 1
                torch.nn.Sigmoid() 
            )
        )
    
    #with forward you define how your model is going to be run, from input to output
    def forward(self, x): #(describe como se ha calculado el output del modelo)
        #named_children() devuelve un iterador sobre los módulos hijos inmediatos de esa clase
        #dando tanto el nombre del módulo como el módulo en sí
        for name, module in self.named_children(): #para cada nombre y módulo de los creados anteriormente
            x = module(x) #output se obtiene alimentando con el input a cada módulo de named_children()
        return x



#ENTRENAMIENTO GAN
# train_discriminator(d_optimizer, discriminator, loss, real_data, fake_data)
def train_discriminator(optimizer, discriminator, loss, real_data, fake_data):

    # Reset gradients (ponemos gradientes a 0)
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data) #le mete lo que detecta como real
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0))) #???
    #pasamos a la retropropagación para mejorar las predicciones
    error_real.backward() ##el error de predicción se propaga hacia atrás a través de la red

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data) #le mete lo que detecta como fake
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward() #retropropagación
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake