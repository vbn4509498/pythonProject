import tensorflow as tf
from tensorflow import keras
from keras import layers
(train_img,train_label),(test_img,test_label)=tf.keras.datasets.mnist.load_data()
train_img = train_img.reshape(60000,28*28)
train_img = train_img.astype('float32')/255
test_img = test_img.reshape(10000,28*28)
test_img = test_img.astype('float32')/255
model = keras.Sequential([
    layers.Dense(512,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']) #acc = accuracy
model.fit(train_img,train_label,epochs=5,batch_size=128)

test_digits = test_img[0:10]
predictions = model.predict (test_digits)
predictions[0] #跑出的值裡面 最接近1的值為預測的值 譬如第1張照片預測值最高的是7，則為7
predictions[0].argmax()# 可抓出儲存值裡最大的值
print(predictions[0][predictions[0].argmax()])
test_label[0]

test_loss,test_acc = model.evaluate(test_img,test_label)
print('test_acc',test_acc)