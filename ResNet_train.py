import tensorflow as tf
import keras
from keras import datasets,optimizers,layers,metrics,Sequential
from ResNet import resnet18
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345) #固定random之後的數值

conv_layers = [ # 5 units of conv + 5 units of maxpool
    # unit 1
    layers.Conv2D(64,kernel_size=[3,3],padding="same",activation='relu'),
    layers.Conv2D(64,kernel_size=[3,3],padding="same",activation='relu'),
    layers.MaxPooling2D(pool_size=[2,2],strides=2,padding='same'),

    # unit 2
    layers.Conv2D(128,kernel_size=[3,3],padding="same",activation='relu'),
    layers.Conv2D(128,kernel_size=[3,3],padding="same",activation='relu'),
    layers.MaxPooling2D(pool_size=[2,2],strides=2,padding='same'),

    # unit 3
    layers.Conv2D(256,kernel_size=[3,3],padding="same",activation='relu'),
    layers.Conv2D(256,kernel_size=[3,3],padding="same",activation='relu'),
    layers.MaxPooling2D(pool_size=[2,2],strides=2,padding='same'),

    # unit 4
    layers.Conv2D(512,kernel_size=[3,3],padding="same",activation='relu'),
    layers.Conv2D(512,kernel_size=[3,3],padding="same",activation='relu'),
    layers.MaxPooling2D(pool_size=[2,2],strides=2,padding='same'),

    # unit 5
    layers.Conv2D(512,kernel_size=[3,3],padding="same",activation='relu'),
    layers.Conv2D(512,kernel_size=[3,3],padding="same",activation='relu'),
    layers.MaxPooling2D(pool_size=[2,2],strides=2,padding='same'),


]

def preprocess(x,y):
    #[0~1]
    x =2*tf.cast(x,dtype=tf.float32)/255.-1
    y =tf.cast(y,dtype=tf.int32)
    return x,y
(x,y),(x_test,y_test)= tf.keras.datasets.cifar100.load_data()
y = tf.squeeze(y,axis=1)
y_test = tf.squeeze(y_test,axis=1)
print(x.shape,y.shape,x_test.shape,y_test.shape)

train_db =tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(64)

test_db =tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(64)

sample = next(iter(train_db))
print('batch:',sample[0].shape,sample[1].shape,tf.reduce_min(sample[0]),tf.reduce_max(sample[0]))


def main():

    #[b,32,32,3] => [b,1,1,512]
    model = resnet18()
    model.build(input_shape=[None,32,32,3])
    model.summary()
    optimizer = optimizers.Adam(learning_rate=1e-3)
    #[1,2]+[3,4] = >[1,2,3,4]

    for epoch in range(50):

        for step ,(x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                #[b,32,32,3] = > [b,100]
                logits = model(x)
                #[b]= >[b,100]
                y_onehot = tf.one_hot(y,depth=100)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))

            if step%100 ==0 :
                print(epoch, step,'loss',float(loss))


        total_num = 0
        total_correct = 0
        for x,y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits,axis=1)
            pred = tf.argmax(prob,axis=1)
            pred =tf.cast(pred,dtype=tf.int32)
            correct = tf.cast(tf.equal(pred,y),dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num+= x.shape[0]
            total_correct+= int(correct)

        acc = total_correct/total_num
        print(epoch,'acc:',acc)



if __name__ == '__main__':
    main()