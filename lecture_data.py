import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


data = np.load('data.npy')
#data_validation = np.load('data_validation.npy')
#data_test = np.load('data_test.npy')

labels = np.load('labels.npy')
data = data.astype(np.float32)
labels = labels.astype(np.float32)

data = (data-data.mean())/data.std()
data_train_base,data_test,labels_train_base,labels_test = train_test_split(data,labels,test_size = 0.1)

dataset_tr = tf.data.Dataset.from_tensor_slices((data_train_base,labels_train_base))
dataset_te = tf.data.Dataset.from_tensor_slices((data_test,labels_test))

batch_size = 8
epoch = 2

for data_batch, labels_batch in dataset_tr.batch(batch_size):

    print(data_batch)
    print(labels_batch)
for data_batch, labels_batch in dataset_te.batch(batch_size):

    print(data_batch)
    print(labels_batch)
