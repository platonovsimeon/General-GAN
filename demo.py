from GAN import GAN
#Creates a new GAN. Parameter howOftenToSave specifies how many epochs there are between savings.
gan =  GAN(height=128,width=128,channel=3,batchSize=20,epochs=100,versionName="New art",dataFolder="data",howOftenToSave=50)
#Trains the model. You need to run this only once. Then you can use the model that was saved during the training.
gan.train()
#Generates a new image. Parameter epochToRestore specifies which save to restore. 
gan.generateNew(epochToRestore=100)
