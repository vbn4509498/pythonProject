import tensorflow as tf
import keras
from keras import datasets,layers,optimizers,metrics,Sequential
import os
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
os.environ['TFF_CPP_MIN_LOG_LEVEL']='2'

(train_data,train_label),(test_data,test_label)=tf.keras.datasets.reuters.load_data(num_words=10000)

def vectorize_sequence(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i , sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results
x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)
#y_train = tf.one_hot(train_label,depth=46)
#y_label = tf.one_hot(test_label,depth=46)
y_train = to_categorical(train_label)
y_test = to_categorical(test_label)

model=Sequential([
    layers.Dense(64,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(46,activation='softmax')
])

model.compile(optimizer='rmsprop',
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history=model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val,y_val)
)





history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend() #增加圖例
plt.show()

plt.clf() #清除圖表
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend() #增加圖例
plt.show()

del model
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(46, activation="sigmoid")
])

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train,y_train,epochs=9,batch_size=512)
results = model.evaluate(x_test,y_test)
