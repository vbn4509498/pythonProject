import tensorflow as tf
import keras
from keras import datasets ,layers ,optimizers,Sequential,metrics
import datetime
import matplotlib.pyplot as plt
import io

def preprocess (x,y):
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.int32)
    return x ,y

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf,format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(),channels=4)
    image = tf.expand_dims(image,0)
    return  image
def image_grid(images):
    figure = plt.figure(figsize= (10,10))
    for i in range(25):
        x=plt.subplot(5,5,i+1)
        x.set_title('name')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i],cmap=plt.cm.binary)
    return figure

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


current_time = datetime.datetime.now().strftime("%Y$m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

sample_img = next(iter(db))[0]
sample_img = sample_img[0]
sample_img = tf.reshape(sample_img, [1, 28, 28, 1])
with summary_writer.as_default():
    tf.summary.image("Training sample",sample_img,step=0)

for step ,(x,y) in enumerate(db):
    with tf.GradientTape() as tape:
        x =tf.reshape(x,(-1,28*28))
        out = network(x)
        y_onehot = tf.one_hot(y,depth=10)
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,out,from_logits=True))

    grads = tape.gradient(loss,network.trainable_variables)
    optimizer.apply_gradients(zip(grads,network.trainable_variables))

    if step%100:
        print(step,"loss",float(loss))
        with summary_writer.as_default():
            tf.summary.scalar('loss', float(loss), step=step)

    if step%500:
        total, total_correct = 0., 0
        for _,(x,y) in enumerate(ds_val):
            x = tf.reshape(x,[-1,28*28])
            out =network(x)
            pred = tf.argmax(out,axis=1)
            pred = tf.cast(pred,dtype=tf.int32)
            correct = tf.equal(pred,y)
            total_correct = tf.reduce_sum(tf.cast(correct,dtype=tf.int32)).numpy()
            total+=x.shape[0]
        print(step,'Evaluate Acc:',total_correct/total)

        val_images = x[:25]
        val_images = tf.reshape(val_images, [-1, 28, 28, 1])

        with summary_writer.as_default():
            tf.summary.scalar('test-acc', float(total_correct/total), step=step)
            tf.summary.image('val-onebyone-images', val_images, max_outputs=25, step=step)
            val_images = tf.reshape(val_images,[-1,28,28])
            figure = image_grid(val_images)
            tf.summary.image('val-images', plot_to_image(figure), step=step)
