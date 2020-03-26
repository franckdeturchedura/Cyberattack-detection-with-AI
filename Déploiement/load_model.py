import tensorflow as tf
import numpy as np
import psutil


loaded_model = tf.keras.models.load_model('model2.h5py',compile=True)


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

    sondes = np.array([freq_cpu,memory_used,bytes_sent,bytes_recv,nbre_process,temp_cpu])
    sondes = np.reshape(sondes,-1,6)
    sondes = sondes.astype(np.float32)
    scaled_sondes = (sondes-sondes.mean())/sondes.std()
    scaled_sondes = np.expand_dims(scaled_sondes,axis=0)
    print(scaled_sondes)
    print(scaled_sondes.shape)
    i=int(i)+1
    pred = loaded_model.predict(scaled_sondes)
    #print(pred)
    #sondes_tenseur = tf.Tensor(scaled_sondes,dtype=tf.float32)

    #predictions = loaded_model.predict(sondes_tenseur)
    #print(predictions)
