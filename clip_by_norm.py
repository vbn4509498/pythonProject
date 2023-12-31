import tensorflow as tf
from keras import optimizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

(x,y),_ = tf.keras.datasets.mnist.load_data()
x = tf.convert_to_tensor(x,dtype=tf.float32)/50.
y = tf.convert_to_tensor(y)
y = tf.one_hot(y,depth=10)
print('x:',x.shape,'y:',y.shape)
train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128).repeat(30)
x,y =next(iter(train_db))
print('sample:',x.shape,y.shape)

def main():
    w1, b1 = tf.Variable(tf.random.truncated_normal([784,512],stddev=0.1)),tf.Variable(tf.zeros([512]))
    w2, b2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    w3, b3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    optimizer = optimizers.SGD(learning_rate= 0.01)

    for step,(x,y) in enumerate(train_db):
        # [b,28,28] = [b,784]
        x = tf.reshape(x,(-1,784))

        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2 @ w3 + b3

            #[b,10] - [b,10]
            loss = tf.square(y-out)
            #[b,10] => [b]
            loss = tf.reduce_mean(loss,axis = 1)
            #[b] => scalar
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
        # print("===before===")
        # for g in grads:
        #     print(tf.norm(g))

        grads , _ = tf.clip_by_global_norm(grads,15)

        # print("===after===")
        # for g in grads:
        #     print(tf.norm(g))
        # update w' = w -lr *grad
        optimizer.apply_gradients(zip(grads,[w1,b1,w2,b2,w3,b3]))


        if step%100 ==0 :
            print(step,'loss:',float(loss))

if __name__ == "__main__":
    main()