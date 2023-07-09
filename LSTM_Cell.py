import tensorflow as tf
import keras
from keras import layers
import os
import numpy as np

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
assert tf.__version__.startswith('2.')
# the most frequence words
total_words = 10000
max_review_len = 80
embedding_len = 100
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.imdb.load_data(num_words=total_words)
# 縮減句子長度至80(80個單字一個句子)
# x_train:[b,80] number
# x_test:[b,80]
x_train = tf.keras.utils.pad_sequences(x_train,maxlen = max_review_len)
x_test = tf.keras.utils.pad_sequences(x_test,maxlen = max_review_len)

batchsz = 128
db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_train = db_train.shuffle(1000).batch(batchsz,drop_remainder=True) #drop_remainder 把最後一個batch drop掉
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.batch(batchsz,drop_remainder=True)
print('x_train shape:',x_train.shape,tf.reduce_max(y_train),tf.reduce_min(y_train))
print('x_test shape:',x_test.shape)


class MyRnn(keras.Model):

  def __init__(self,units):
    super(MyRnn,self).__init__()
    #[b,64]
    self.state=[tf.zeros([batchsz,units]),tf.zeros([batchsz,units])]
    self.state1=[tf.zeros([batchsz,units]),tf.zeros([batchsz,units])]
    # transform text to embedding representation
    # [b,80] => [b,80,100]
    self.embedding = layers.Embedding(total_words,embedding_len,input_length=max_review_len)

    #[b,80,100] => hid_dim:64
    #RNN:cell1,cell2,cell3
    #SimpleRnn
    # self.rnn_cell = layers.SimpleRNNCell(units,dropout=0.2)
    # self.rnn_cell1 = layers.SimpleRNNCell(units,dropout=0.2)
    self.rnn_cell = layers.LSTMCell(units,dropout=0.5)
    self.rnn_cell1 = layers.LSTMCell(units, dropout=0.5)

    #fc [b,80,100]=>[b,64] = [b,1]
    self.outlayer = layers.Dense(1)


  def call(self,inputs,training=None):
    """
    net(x)或net(x,training=True) => train mode
    net(x,training=False) =>test mode
    :param inputs:[b,80]
    :param training:
    :return:
    """
    #[b,80]
    x = inputs
    # embedding :[b,80] => [b,80,100]
    x=self.embedding(x)
    # rnncell compute
    # [b,80,100] => [b,64]
    state = self.state
    state1 = self.state1
    for word in tf.unstack(x,axis=1): # word:[b,100 ]
      # h1 = x*Wxh + h0 * Whh h1=state1=out
      out,state = self.rnn_cell(word,state,training)
      out1,state1 = self.rnn_cell1(out,state1,training)
    # out:[b,64] => [b,1]
    x = self.outlayer(out1)
    # p(y is pos|x)
    prob = tf.sigmoid(x)
    return prob


def main():
  units = 64
  epochs =4
  import time
  t0 = time.time()
  model = MyRnn(units)

  model.compile( optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
          loss = tf.losses.BinaryCrossentropy(),
          metrics = ['accuracy'])
  model.fit(db_train,epochs=epochs,validation_data=db_test)
  model.evaluate(db_test)
  t1 = time.time()
  print('total time cost:',t1-t0)
if __name__ == "__main__":
  main()