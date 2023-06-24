import tensorflow as tf
import keras
from keras import layers ,Sequential,optimizers,metrics

import os

os.environ['TFF_CPP_MIN_LOG_LEVEL']='2'
def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y
batchsz = 128
(x,y),(x_val,y_val) = tf.keras.datasets.fashion_mnist.load_data()
print('datasets',x.shape,y.shape,x.min(),y.min())

db=tf.data.Dataset.from_tensor_slices((x,y))
db =db.map(preprocess).shuffle(60000).batch(batchsz).repeat(10)

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
optimizer =optimizers.Adam(learning_rate=0.01)

acc_meter = keras.metrics.Accuracy()
loss_meter = keras.metrics.Mean()
def main():

    for step,(x,y) in enumerate (db):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28 * 28))
            out = network(x)
            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))
            loss_meter.update_state(loss)
        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))

        if step % 100 == 0:
            print(step,'loss',loss_meter.result().numpy())
            loss_meter.reset_states()
        if step % 500 == 0:
            total, total_correct = 0., 0
            acc_meter.reset_states()
            for step, (x, y) in enumerate(ds_val):
                x = tf.reshape(x, [-1, 28 * 28])
                out = network(x)
                pred = tf.argmax(out, axis=1)
                pred = tf.cast(pred, dtype=tf.int32)
                correct = tf.equal(pred, y)
                total_correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]
                acc_meter.update_state(y,pred)
            print(step, 'Evaluate Acc:', total_correct / total,acc_meter.result().numpy())







if __name__ == "__main__":
    main()