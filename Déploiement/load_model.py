import tensorflow as tf
import numpy as np
import psutil
import time



loaded_model = tf.keras.models.load_model('model2.h5py',compile=True)
classes = ['normal','attack']
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
    time.sleep(0.2)
    pred = loaded_model.predict(scaled_sondes)
    answer = classes[pred.argmax()]
    print(pred)
    print("Answer : ",answer)
