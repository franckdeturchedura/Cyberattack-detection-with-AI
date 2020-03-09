import tensorflow as tf
import numpy as np
from .data_process import labels_train

class DenseModelV1(tf.keras.Model):#on considère que le réseau n'est pas récurrent

  def __init__(self):
    super(DenseModelV1,self).__init__()
    self.dense1 = tf.keras.layers.Dense(8,name="dense1")
    self.activation = tf.keras.layers.LeakyReLU(alpha=0.2,name="activation")#on utilise leakyrelu pour éviter que des neurones "meurent" car ils ne sont jamais activés
    self.dense2 = tf.keras.layers.Dense(16,name="dense2")
    self.dense3 = tf.keras.layers.Dense(8,name="dense3")
    self.out = tf.keras.layers.Dense(2,activation="softmax",name="out")

  def call(self,data):
    dense1 = self.dense1(data)
    activation1 = self.activation(dense1)
    dense2 = self.dense2(activation1)
    activation2 = self.activation(dense2)
    dense3 = self.dense3(activation2)
    out = self.out(dense3)

    return out


#model = DenseModelV1()




@tf.function
def train_step(data,label,model):
  with tf.GradientTape() as tape:
    prediction = model(data)
    loss = loss_object(label,prediction)#on utilise l'objet loss définit plus haut (ici CategoricalCrossentropy)
  gradients = tape.gradient(loss,model.trainable_variables)#calcule le gradient avec la perte et la prediction ci dessus
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))#on utilise l'optimizer définit plus haut (ici Adam)
  train_loss(loss)#On utilise la metric de perte définie plus haut
  train_accuracy(label,prediction)#on utilise la metric de moyenne définie plus haut


#ici la fonction de validation
@tf.function#on convertit automatiquement en mode graphe pour avoir de meilleures performances
def validation_step(data,label,model):
  prediction = model(data)
  t_loss= loss_object(label,prediction)
  validation_loss(t_loss)
  validation_accuracy(label,prediction)

#on va définir la fonction de cout et l'optimizer selon notre problème
loss_object = tf.keras.losses.CategoricalCrossentropy()#pas Sparse car one hot encoding
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)#descente de gradient optimisée #learning rate dépendra du nombre de données

#On définit nos metrics pour voir si le réseau se comporte bien
#Pour le perte
train_loss = tf.keras.metrics.Mean(name='train_loss')
validation_loss = tf.keras.metrics.Mean(name='validation_loss')

#Pour la moyenne
#Pas de sparse car one hot encoding
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
validation_accuracy = tf.keras.metrics.CategoricalAccuracy(name='validation_accuracy')


def train_network(model,train_dataset,validation_dataset,epoch,batch_size):

    b=0
    epoch = epoch
    print_train_loss=tf.keras.metrics.Mean(name='print_train_loss')
    print_validation_loss=tf.keras.metrics.Mean(name='print_validation_loss')
    print_train_accuracy= tf.keras.metrics.CategoricalAccuracy(name='print_train_accuracy')
    print_validation_accuracy= tf.keras.metrics.CategoricalAccuracy(name='print_validation_accuracy')

    loss_tra = np.array([])
    loss_val = np.array([])
    acc_tra = np.array([])
    acc_val = np.array([])

    for epoch in range(epoch):
      #training set
      b=0

      for data_batch, labels_batch in train_dataset.batch(batch_size):
        train_step(data_batch,labels_batch,model)
        template = 'Batch {}/{}, Loss: {}, Accuracy: {} \n'
        print(template.format(b,len(labels_train),train_loss.result(),train_accuracy.result()*100),end="")
        b += batch_size
      #validation set
      for data_batch,labels_batch in validation_dataset.batch(batch_size):
        validation_step(data_batch,labels_batch,model)
      template = 'Epoch {}, Validation Loss : {}, Validation accuracy : {} \n'
      print(template.format(epoch+1,validation_loss.result(),validation_accuracy.result()*100))
      b =0
      #print_train_loss.update_state(train_loss.result())
      #print_validation_loss.update_state(validation_loss.result())
      #print_train_accuracy.update_state(train_accuracy.result())
      #print_validation_accuracy.update_state(validation_accuracy.result())
      loss_tra = np.append(loss_tra,np.asarray(float(train_loss.result())))
      loss_val = np.append(loss_val,np.asarray(float(validation_loss.result())))
      acc_val = np.append(loss_val,np.asarray(float(validation_accuracy.result())))
      acc_tra = np.append(loss_val,np.asarray(float(train_accuracy.result())))


      validation_loss.reset_states()
      train_loss.reset_states()
      validation_accuracy.reset_states()
      train_accuracy.reset_states()

    return model,loss_tra,loss_val,acc_val,acc_tra
