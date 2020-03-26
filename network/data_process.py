#à la fin o souhaite avoir une base de données clean prête à être donnée au network
import psutil
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time


def get_temp_cpu_ind():
    temp = psutil.sensors_temperatures()
    temp = str(temp)
    ind=0
    for i in range(0,len(temp)-1):
        if temp[i]=="c" and temp[i+1]=="u" and temp[i+2]:
            return ind
        ind = ind+1
    return None

def get_temp_cpu():
    ind = get_temp_cpu_ind()
    temp = psutil.sensors_temperatures()
    temp = str(temp)
    return float(temp[ind+8:ind+10])




def get_data():
    freq_cpu = 0
    while(freq_cpu==0):
        freq_cpu=psutil.cpu_percent()
    temp_cpu = get_temp_cpu()
    svmem = psutil.virtual_memory()
    memory_used = get_size(svmem.used)
    bytes_sent = psutil.net_io_counters().bytes_sent
    bytes_recv = psutil.net_io_counters().bytes_recv
    #to see processes
    nbre_process = 0

    for i in psutil.process_iter():
        nbre_process = nbre_process+1
    return freq_cpu,memory_used,bytes_sent,bytes_recv,nbre_process,temp_cpu

def create_data():
    freq_cpu,memory_used,bytes_sent,bytes_recv,nbre_process,temp_cpu = get_data()
    freq_cpu = float(freq_cpu)
    bytes_sent = float(bytes_sent)
    bytes_recv = float(bytes_recv)
    memory_used = float(memory_used[0:4])
    nbre_process = float(nbre_process)
    temp_cpu = float(temp_cpu)
    input_data = np.array([freq_cpu,memory_used,bytes_sent,bytes_recv,nbre_process,temp_cpu])
    input_data = np.reshape(input_data,-1,6)
    #print(input_data[3])
    return input_data

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


final_data = np.array([[[]]])
final_labels = np.array([[[]]])



batch_size = 2
epoch = 2

print("Quel type d'attaque ? 0 :None, 1 : Attaque, ...,100 : Stop")
attak = 0

while(attak != 100):

    attak = input()
    attak = int(attak ) #0 pas d'attaque, 1 attaque1, 2 attaque2,...
    if attak ==0:

        for i in range(0,epoch):
          test_data = np.array([[]])
          test_labels = np.array([[]])
          #création des batchs
          for i in range(0,batch_size):
              input_data = create_data()
              input_data = np.array([input_data])
              input_data = np.reshape(np.array([input_data]),(-1,6))
             #print(input_data.shape)

              test_data = np.append(test_data,input_data)
              test_data = np.reshape(test_data,(-1,6))
              test_labels = np.append(test_labels,np.array([1,0]))
              test_labels = np.reshape(test_labels,(-1,2))
              print(input_data)
              print(test_labels[i])
              time.sleep(0.3)
          final_data = np.append(final_data,test_data)
          final_data = np.reshape(final_data,(-1,1,6))
          final_labels = np.append(final_labels,test_labels)
          final_labels = np.reshape(final_labels,(-1,1,2))
        print("Quel type d'attaque ? 0 :None, 1 : Attaque, ...,100 : Stop")



    elif attak==1:
        for i in range(0,epoch):
          test_data = np.array([[]])
          test_labels = np.array([[]])
          #création des batchs
          for i in range(0,batch_size):
              input_data = create_data()
              input_data = np.array([input_data])
              input_data = np.reshape(np.array([input_data]),(-1,6))
             #print(input_data.shape)

              test_data = np.append(test_data,input_data)
              test_data = np.reshape(test_data,(-1,6))
              test_labels = np.append(test_labels,np.array([0,1]))#change les labels ici
              test_labels = np.reshape(test_labels,(-1,2))
              print(test_data.shape)
              print(test_labels[i])

              time.sleep(0.3)
          final_data = np.append(final_data,test_data)
          final_data = np.reshape(final_data,(-1,1,6))
          final_labels = np.append(final_labels,test_labels)
          final_labels = np.reshape(final_labels,(-1,1,2))
        print("Quel type d'attaque ? 0 :None, 1 : Attaque, ...,100 : Stop")






data_train_base,data_test,labels_train_base,labels_test = train_test_split(final_data,final_labels,test_size = 0.2)
data_train,data_validation,labels_train,labels_validation = train_test_split(data_train_base,labels_train_base,test_size=0.3)

final_data = final_data.astype(np.float32)
final_labels = final_labels.astype(np.float32)
#scaled_data_test = (data_test-data_test.mean())/data_test.std()

scaled_data_train = (data_train-data_train.mean())/data_train.std()
scaled_data_validation = (data_validation-data_validation.mean())/data_validation.std()
scaled_data_test = (data_test-data_test.mean())/data_test.std()

train_dataset = tf.data.Dataset.from_tensor_slices((scaled_data_train,labels_train))
test_dataset = tf.data.Dataset.from_tensor_slices((scaled_data_test,labels_test))
validation_dataset = tf.data.Dataset.from_tensor_slices((scaled_data_validation,labels_validation))



for data_batch, labels_batch in train_dataset.batch(batch_size):

    print(data_batch)
    print(labels_batch)
