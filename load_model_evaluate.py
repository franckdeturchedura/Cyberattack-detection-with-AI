import tensorflow as tf
import numpy as np
from lecture_data import dataset_tr,dataset_te
from network.data_process import final_data,final_labels,scaled_data_test
import psutil


loaded_model = tf.keras.models.load_model('model2.h5py',compile=True)


print(scaled_data_test.shape)
pred = loaded_model.predict(scaled_data_test[0:1])#ça marche sah quel plaisir
print(pred)

"""
loaded_model.compile(
  loss="categorical_crossentropy",
  optimizer="adam",
  metrics=["accuracy"])
loaded_model.evaluate(final_data,final_labels)#ça marche bien
"""
