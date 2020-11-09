import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv


pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20


# reading the data
trainVal = pd.read_csv('labeledTrainData.tsv', delimiter="\t", header = 0,  quoting = 3 )
test = pd.read_csv('testData.tsv', delimiter="\t", header = 0,  quoting = 3 )
train_un = pd.read_csv('unlabeledTrainData.tsv', delimiter="\t", header = 0,  quoting = 3 )

# hyperparameters
split_size = 0.2
number_words_token = 100000
max_len_pad = 50
embedding_dim = 16


test.head()

trainVal.drop(columns='id')

trainVal["review"][0]

# Stemming or lemmatization ? Here, I selected the first one.
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')    # Download stopwords: a - an - the - and - or ...
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer   # having the roots of all words: e.g. loved -> love

#----------------------
# Most data cleaning is done with a tokenizer in keras. 
# But, I am going to use it anyway! 
def clean_text(dataset):
  a = []
  for i in range(dataset.shape[0]):   # cleaning a sentence for each reviewer
      # everything not included in a-zA-Z are replaced by space
      review = re.sub('[^a-zA-Z]', ' ', dataset['review'][i]) 
      # now, a sentence is an object that has attributes. 
      # We use that to transform all Upper case letters to lower case letters
      review = review.lower()
      # We split all words in a sentence. Consider it as we transform it into a list of words
      review = review.split() 
      # Creating an object of PorterStemmer class
      ps = PorterStemmer()
      word_element = []
      sw = stopwords.words('english')
      # why we remove wasn, not wasn't: Because, everything not included in a-zA-Z are replaced by space
      # so, wasn't -> wasn t
      remove_stop_words = ["not", "wouldn", "won", "weren", "hasn", "don", "isn", "aren", "wasn", "didn"]
      for j in remove_stop_words:
          sw.remove(j)
      for word in review:
          # determining whether a word is in stopwords or not (e.g. if word = 'an', we do not consider it)
          if not word in set(sw):
              # the root of a word (e.g. loved -> love)
              word_element.append(ps.stem(word))
      review = word_element # this is a list of words
      review = ' '.join(review) # now, we convert it to a string where words are separated with space
      # dataset['review'][i] = review
      a.append(review )
  return a
      #if i==1:
      #   print(corpus)



trainVal_sen = clean_text(trainVal)
test_sen = clean_text(test)
train_unlabeled_sen = clean_text(train_un)



trainVal_label =trainVal['sentiment'].values

# splitting
split = int(len(trainVal_sen) * split_size)
val_sen = trainVal_sen[:split]
train_sen = trainVal_sen[split:]

val_label = trainVal_label[:split]
train_label = trainVal_label[split:]

train_label
len(train_sen)
train_sen[6]



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=number_words_token, oov_token="<OOV>" )

# each word in train_unlabeled is mapped to a code number
# here we fit it on train_unlabeled_sen, since it has much more words than to train_sen
tokenizer.fit_on_texts(train_unlabeled_sen) 

# for each sentence in a train_unlabeled_sen list, we construct a sequence of code numbers based on 
# the given code number for each word
train_seq = tokenizer.texts_to_sequences(train_sen) 

# zero padding: when the number of words is less than a max_length
train_seq_pad = pad_sequences(train_seq, maxlen = max_len_pad, padding = 'post' , truncating="post")

val_seq = tokenizer.texts_to_sequences(val_sen)
val_seq_pad = pad_sequences(val_seq, maxlen = max_len_pad, padding = 'post' , truncating="post")

test_seq = tokenizer.texts_to_sequences(test_sen)
test_seq_pad = pad_sequences(test_seq, maxlen = max_len_pad, padding = 'post' , truncating="post")

print(train_seq_pad.shape)
train_seq_pad[1:3,:]

# Neural network model with Conv1D and bidirectional LSTM layers

from tensorflow.keras.regularizers import l2

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(number_words_token, embedding_dim, input_length=max_len_pad),
    tf.keras.layers.Dropout(0.5),
   # tf.keras.layers.Conv1D(filters = 16, kernel_size = 5, activation='relu'),
   # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 32, return_sequences = True,
            recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 32, 
            recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


train_padd = np.array(train_seq_pad)
train_label = np.array(train_label)

val_padd = np.array(val_seq_pad)
val_label = np.array(val_label)

test_padd = np.array(test_seq_pad)
#test_label = np.array(test_labels)

number_epochs = 40
ModelHistory = model.fit(x = train_padd, y = train_label, epochs=number_epochs, 
                    validation_data=(val_padd, val_label), verbose=2)

# accuracy vs iterations

TrainAcc = ModelHistory.history['accuracy']
ValAcc = ModelHistory.history['val_accuracy']
TrainLoss = ModelHistory.history['loss']
ValLoss = ModelHistory.history['val_loss']

Nepochs = range(len(TrainAcc))

plt.plot(Nepochs, TrainAcc, 'k', label='Training')
plt.plot(Nepochs, ValAcc, 'r', label='Validation')
plt.title('accuracy - Training & validation')
plt.ylabel('Accuracy[%]')
plt.xlabel('Iteration')
plt.legend(loc=0)
plt.figure()


plt.show()


# Based on the figure, there is overfitting in a trained algorithm. 
# however, I tried different ways to overcome this difficulty, such as:
# drop out, regularization, tuning hyperparameters, different kind of layers
# but it did not work. Maybe we should use sub words or letters. 
# Or maybe using Embedding vectors obtained from a much larger data set.

# predicting for the test set
results = model.predict(test_padd)

results_binary = np.zeros((len(results)))
for i in range(len(results_binary)):
  if results[i]>= 0.5:
    results_binary[i] = int(1)
  else:
    results_binary[i] = int(0)
    

results_binary

test = pd.read_csv('testData.tsv', delimiter="\t", header = 0)
test['id'].values

results = pd.Series(results_binary,name="sentiment")
submission = pd.concat([pd.Series(test['id'].values,name = "id"),results],axis = 1)
submission.to_csv("submission.csv",index=False)
