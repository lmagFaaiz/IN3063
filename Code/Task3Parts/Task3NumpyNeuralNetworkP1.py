# Import libraries
import numpy as np
import time
import matplotlib.pyplot as plt

from torchvision import transforms, datasets

#Get train and test data sets from torchvision.datsets and transform to tensor form
train_data = datasets.FashionMNIST('./FMNIST', train=True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
test_data = datasets.FashionMNIST('./FMNIST', train=False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

#Split data into data and targets for pre-processing
#X is images, Y is labels
trainX = train_data.data
valX = test_data.data
trainY = train_data.targets
valY = test_data.targets


#preprocess data by scaling to range from 0 to 1
#this also changes img to grayscale/255
trainX = trainX / 255.0
valX = valX / 255.0

#store class names for later
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#checking details
print(trainX.shape)
print(len(trainY))
print(trainY)
print(valX.shape)
print(len(valY))

#verify format and check first 15 images
plt.figure(figsize=(6,4))
for i in range(15):
    plt.subplot(3,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainX[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[trainY[i]])
plt.show()