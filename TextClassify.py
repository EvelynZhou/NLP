import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

imdb=keras.datasets.imdb

(train_data,train_labels),(test_data, test_labels)=imdb.load_data(num_words=10000)

print("Training entries:{}, labels:{}".format(len(train_data),len(train_labels)))

print(train_data[0])

word_index=imdb.get_word_index()

word_index={k:(v+3) for k,v in word_index.items()}
word_index['<PAD>']=0
word_index['<START>']=1
word_index['<UNK>']=2
word_index['<UNUSED>']=3

reverse_word_index=dict([(value,key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i) for i in text])

str= decode_review(train_data[0])

#长度标准化
train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=256,dtype='int32',padding='post',value=word_index['<PAD>'])
test_data = keras.preprocessing.sequence.pad_sequences(test_data, maxlen=256,dtype='int32',padding='post',value=word_index['<PAD>'])

print(train_data[0])

#构建模型
vocab_size=10000
model=keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation=tf.nn.relu))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),loss='binary_crossentropy',metrics=['accuracy'])

x_val=train_data[:10000]
partial_x_train=train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history=model.fit(partial_x_train,partial_y_train,epochs=40,batch_size=512,validation_data=(x_val,y_val),verbose=1)
results=model.evaluate(test_data, test_labels)

#可视化

import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,loss,'bo',label='training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('training and validation loss')
plt.xlabel('Epochs')
plt.legend('loss')

plt.show()


