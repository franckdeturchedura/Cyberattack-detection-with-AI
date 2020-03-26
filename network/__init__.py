import tensorflow as tf
#print(tf.__version__)
from .data_process import train_dataset,validation_dataset,test_dataset,scaled_data_train
from .network_model import train_network,DenseModelV1
from .viewdata import train_data_viewer,validation_data_viewer,test_data_viewer,plot_acc,plot_loss
from lecture_data import  dataset_tr,dataset_te

print(tf.__version__)

model = DenseModelV1()

train_dataset = train_dataset
validation_dataset = validation_dataset
test_dataset = test_dataset

model.predict(scaled_data_train[0:1])

#train_data_viewer()
print(model.summary())



#model,loss_tra,loss_val,acc_val,acc_tra=train_network(model,dataset_tr,dataset_te,1000,32)
#plot_acc(acc_tra,acc_val)
#plot_loss(loss_tra,loss_val)

tf.keras.models.save_model(model,'model.h5py')
