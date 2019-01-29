# General-GAN, a general purpose GAN. 

# What is this project about?
This project contains a general purpose GAN, that can learn to generate new images from any dataset that contains images.

# What is a GAN?
GAN stands for generative adversarial network. It is a neural network that can learn to generate new data based on its training data. 

# Usage
from GAN import GAN
#Creates a new GAN. Parameter howOftenToSave specifies how many epochs there are between savings.
gan =  GAN(height=128,width=128,channel=3,batchSize=20,epochs=100,versionName="New art",dataFolder="data",howOftenToSave=50)
#Trains the model. You need to run this only once. Then you can use the model that was saved during the training.
gan.train()
#Generates a new image. Parameter epochToRestore specifies which save to restore. 
gan.generateNew(epochToRestore=100)
