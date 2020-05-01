import os
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Bidirectional
from keras.datasets import imdb

max_features = 60000 # This refers to the number of input vertices (or features) in the model
maxlen = 80 # This is the maximum number of words
batch_size = 100 # Number of training samples executed prior to doing back prop

print('Loading data...')
# This function loads data from the imdb data set.  This function's full / documentation and
# usage can be found in the Keras documentation.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
# The following two statements pad /  truncate the reviews so that they are of all equal length.
# This is what allows for uniform vectors for input into the model.
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
# All this command does is tell Keras to interpret the layers as a linear addition.
# e.g. the first model.add() is the first layer, the second model.add() is the second and so on so forth.
model = Sequential()
# Embedding is an NLP model that has the goal of grouping similar words. For example good and great are similar
# This layer is trying to mathmatically model this in that way.
# The second input means there are n categories for the words to be filtered too.  
model.add(Embedding(max_features, 128))
model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.3)))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Logic for saving model
checkpoint_dir = os.getcwd()
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.getcwd(),save_weights_only=True,verbose=1)

print("Estimated Size of Model " + str((model.count_params()* 4)/1000) + "kB")
print('Train...')
if False:
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=1,
            validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
            batch_size=batch_size, callbacks=[cp_callback])