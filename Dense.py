import keras
import tensorflow as tf

x = tf.random.normal([2,3])

model = keras.Sequential([
        keras.layers.Dense(2,activation="relu"),
        keras.layers.Dense(2,activation="relu"),
        keras.layers.Dense(2),
])
model.build(input_shape=[None,3])
model.summary()

for p in model.trainable_variables:
    print(p.name,p.shape)

# import tensorflow as tf
# gpu_device_name = tf.test.gpu_device_name()
# print(gpu_device_name)
# print(tf.config.list_physical_devices('GPU'))