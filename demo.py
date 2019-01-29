from GAN import GAN
#Creates a new GAN. Parameter howOftenToSave specifies how many epochs there are between savings.
gan =  GAN(height=128,width=128,channel=3,batchSize=20,epochs=100,versionName="New art",dataFolder="data",howOftenToSave=50)
#Trains the model. This step is not necessary if you have a trained model already in the folder "model/versionName".
gan.train()
#Generates a new image. Parameter epochToRestore specifies which save to restore. 
gan.generateNew(epochToRestore=100)
