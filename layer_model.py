import tensorflow as tf
import keras
from keras import layers ,Sequential,optimizers,metrics

import os


os.environ['TFF_CPP_MIN_LOG_LEVEL']='2'
def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255.
    x = tf.reshape(x,[28*28])
    y = tf.cast(y,dtype=tf.int32)
    y = tf.one_hot(y,depth=10)
    return x,y
batchsz = 128
(x,y),(x_val,y_val) = tf.keras.datasets.fashion_mnist.load_data()
print('datasets',x.shape,y.shape,x.min(),x.max())


db=tf.data.Dataset.from_tensor_slices((x,y))
db =db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val=tf.data.Dataset.from_tensor_slices((x_val,y_val))
ds_val =ds_val.map(preprocess).batch(batchsz,drop_remainder=True)

network =Sequential([layers.Dense(256,activation='relu'),#[b,784] => [b,256]
    layers.Dense(128,activation='relu'),#[b,256] => [b,128]
    layers.Dense(64,activation='relu'),#[b,128] => [b,64]
    layers.Dense(32,activation='relu'),#[b,64] => [b,32]
    layers.Dense(10)#[b,32] => [b,10] # 330 = 32*10+10
])
network.build(input_shape=(None,28*28))
network.summary()

class MyDense(layers.Layer):
    def __init__(self,input_dim,output_dim):
        super(MyDense,self).__init__()

        self.kernal =self.add_weight('w',[input_dim,output_dim])
        self.bias =self.add_weight('b',[output_dim])

    def call(self, inputs, training=None):
        out =inputs @ self.kernal + self.bias

        return out
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.fc1 = MyDense(28*28,256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)
    def call(self, inputs, training=None):

        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)

        return x

network=MyModel()

network.compile(optimizer=optimizers.Adam(learning_rate=0.01),
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['acc']) #acc = accuracy
network.fit(db,epochs=5,validation_data=ds_val,validation_freq=2)

network.evaluate(ds_val)

sample = next(iter(ds_val))
x=sample[0]
y=sample[1]
pred = network.predict(x)

