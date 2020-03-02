import numpy as np

data_train = np.load('data_train.npy')
data_validation = np.load('data_validation.npy')
data_test = np.load('data_test.npy')

print("Data train : \n\n" ,data_train)
print("Data validation :\n\n ",data_validation)
print("Data test : \n\n",data_test)
