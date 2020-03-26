import tensorflow as tf
import numpy as np
from lecture_data import dataset_tr,dataset_te
from network.data_process import final_data,final_labels,scaled_data_test
import psutil


loaded_model = tf.keras.models.load_model('model2.h5py',compile=True)


print(scaled_data_test[0:1].shape)
pred = loaded_model.predict(scaled_data_test[0:1])#ça marche sah quel plaisir
print(pred)

"""
loaded_model.compile(
  loss="categorical_crossentropy",
  optimizer="adam",
  metrics=["accuracy"])
loaded_model.evaluate(final_data,final_labels)#ça marche bien
"""

i=0
while(i<100):
    freq_cpu = 0
    while(freq_cpu==0):
        freq_cpu=psutil.cpu_percent()
    temp_cpu = 50
    svmem = psutil.virtual_memory()[2]
    memory_used = svmem
    bytes_sent = psutil.net_io_counters().bytes_sent
    bytes_recv = psutil.net_io_counters().bytes_recv
    #to see processes
    nbre_process = 0

    for j in psutil.process_iter():
        nbre_process = nbre_process+1

    sondes = np.array([float(freq_cpu),float(memory_used),float(bytes_sent),float(bytes_recv),float(nbre_process),float(temp_cpu)])
    sondes = np.reshape(sondes,-1,6)
    sondes = sondes.astype(np.float32)
    scaled_sondes = (sondes-sondes.mean())/sondes.std()
    scaled_sondes = np.expand_dims(scaled_sondes,axis=0)
    scaled_sondes = np.expand_dims(scaled_sondes,axis=1)

    print(scaled_sondes)
    print(scaled_sondes.shape)
    i=int(i)+1
    pred = loaded_model.predict(scaled_sondes)
    print(pred)
