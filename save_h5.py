import tensorflow as tf
from .app import model

tf.keras.models.save_model(model,'model.h5py')
