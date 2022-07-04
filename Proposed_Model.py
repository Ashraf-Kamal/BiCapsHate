import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Input, Embedding, Bidirectional, TimeDistributed, merge, concatenate, GRU
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import keras_metrics
import keras.backend as K
from keras.models import Sequential
import tweepy
import os
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras_self_attention import SeqSelfAttention
import Capsule_Class as Caps
import Attention_Class as Attention
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import roc_curve, auc
print (keras.__version__)


import random as rn
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
rn.seed(1254)
tf.set_random_seed(89)
from keras import backend as K
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

df1 = pd.read_csv(r'Linguistic_Features\FV_Results\Davidson_Balanced.csv',delimiter=',',encoding='latin-1')
#df1 = pd.read_csv(r'Linguistic_Features\FV_Results\Founta_Balanced.csv',delimiter=',',encoding='latin-1')
#df1 = pd.read_csv(r'Linguistic_Features\FV_Results\Gao_Balanced.csv',delimiter=',',encoding='latin-1')
#df1 = pd.read_csv(r'Linguistic_Features\FV_Results\Roy_Balanced.csv',delimiter=',',encoding='latin-1')
#df1 = pd.read_csv(r'Linguistic_Features\FV_Results\HateXplain_Balanced.csv',delimiter=',',encoding='latin-1')

#df1 = pd.read_csv(r'Linguistic_Features\FV_Results\Davidson_Unbalanced.csv',delimiter=',',encoding='latin-1')
#df1 = pd.read_csv(r'Linguistic_Features\FV_Results\Founta_Unbalanced.csv',delimiter=',',encoding='latin-1')
#df1 = pd.read_csv(r'Linguistic_Features\FV_Results\Gao_Unbalanced.csv',delimiter=',',encoding='latin-1')
#df1 = pd.read_csv(r'Linguistic_Features\FV_Results\Roy_Unbalanced.csv',delimiter=',',encoding='latin-1')
#df1 = pd.read_csv(r'Linguistic_Features\FV_Results\HateXplain_Unbalanced.csv',delimiter=',',encoding='latin-1')


print (df1)

properties = list(df1.columns.values)
properties.remove('Label')
print(properties)
X1 = df1[properties]
Y1 = df1['Label']

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)
print (X1_train)
print (Y1_train)


df = pd.read_csv(r'Final_Datasets\Davidson_17\Davidson_Balanced.csv',delimiter=',',encoding='latin-1')
#df = pd.read_csv(r'Final_Datasets\Founta_18\Founta_Balanced.csv',delimiter=',',encoding='latin-1')
#df = pd.read_csv(r'Final_Datasets\Gao_17\Gao_Balanced.csv',delimiter=',',encoding='latin-1')
#df = pd.read_csv(r'Final_Datasets\Roy_20\Roy_Balanced.csv',delimiter=',',encoding='latin-1')
#df = pd.read_csv(r'Final_Datasets\HateXplain\HateXplain_Bi_Balance.csv',delimiter=',',encoding='latin-1')

#df = pd.read_csv(r'Final_Datasets\Davidson_17\Davidson_Unbalanced.csv',delimiter=',',encoding='latin-1')
#df = pd.read_csv(r'Final_Datasets\Founta_18\Founta_Unbalanced.csv',delimiter=',',encoding='latin-1')
#df = pd.read_csv(r'Final_Datasets\Gao_17\Gao_Unbalanced.csv',delimiter=',',encoding='latin-1')
#df = pd.read_csv(r'Final_Datasets\Roy_20\Roy_Unbalanced.csv',delimiter=',',encoding='latin-1')
#df = pd.read_csv(r'Final_Datasets\HateXplain\HateXplain_Bi_Imbalance.csv',delimiter=',',encoding='latin-1')

df=df.dropna()

Routings = 3
Num_capsule = 5
Dim_capsule = 8

X = df.Text
Y = df.Label
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)

train_msg, test_msg, train_labels, test_labels = train_test_split(X, Y,test_size=0.2, random_state = 42)

max_words =5000
max_len = 20

tokenizer= Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_msg)

training_sequences = tokenizer.texts_to_sequences(train_msg)
training_padded = sequence.pad_sequences(training_sequences,maxlen=max_len)

test_sequences = tokenizer.texts_to_sequences(test_msg)
testing_padded = sequence.pad_sequences(test_sequences,maxlen=max_len)

embeddings_index = dict()
f = open('F:\Glove\glove.twitter.27B.100d.txt',encoding="utf8")
for line in f:    
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
print (len(embeddings_index))
f.close()
embedding_matrix = np.zeros((max_words, 100))
for word, index in tokenizer.word_index.items():
    if index > max_words - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
print (embedding_matrix)


main_input = Input(shape=(20,), name='main_input')
aux_input = Input(shape=(26,), name='aux_vector_input')


embedding=Embedding(max_words, 100, input_length=max_len, weights=[embedding_matrix], trainable=False)(main_input)

LSTM1= LSTM(128, return_sequences=True)(embedding)
LSTM1_Drop = Dropout(0.5)(LSTM1)
print('LSTM',LSTM1_Drop.shape)
caps1 = Caps.Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(LSTM1_Drop)

LSTM2= LSTM(128, return_sequences=True, go_backwards=True)(embedding)
LSTM2_Drop = Dropout(0.5)(LSTM2)

caps2 = Caps.Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(LSTM2_Drop)

merged_capsules = concatenate([caps1, caps2])
attention= Attention.attention()(merged_capsules)
merged_features = concatenate([attention,aux_input])
Final_output = Dense(1, activation='sigmoid', trainable=True)(merged_features)
model = Model(inputs=[main_input,aux_input], outputs=Final_output)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', precision_m, recall_m, f1_m])
model.summary()

#keras.utils.plot_model(model, "Layer.png", show_shapes=True)

hist =model.fit([training_padded,X1_train],train_labels, validation_data=([testing_padded,X1_test], test_labels), batch_size=128, epochs=50, verbose=2)
accr = model.evaluate([testing_padded,X1_test], test_labels, verbose=0)
#print('Testing set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f} \n  Precision: {:0.3f} \n  Recall: {:0.3f} \n  F-score: {:0.3f}'.format(accr[0],accr[1],accr[2],accr[3],accr[4]))

print ('Training loss:', np.mean(hist.history['loss']))
print ('Training accuracy:', np.mean(hist.history['acc']))
print ('Training precision:', np.mean(hist.history['precision_m']))
print ('Training recall:', np.mean(hist.history['recall_m']))
print ('Training f-score:', np.mean(hist.history['f1_m']))

print ('**************')

print ('Testing loss:', np.mean(hist.history['val_loss']))
print ('Testing accuracy:', np.mean(hist.history['val_acc']))
print ('Testing precision:', np.mean(hist.history['val_precision_m']))
print ('Testing recall:', np.mean(hist.history['val_recall_m']))
print ('Testing f-score:', np.mean(hist.history['val_f1_m']))

model.save('Model_Result\model.Proposed_Model_Davidson_Balanced')
#model.save('Model_Result\model.Proposed_Model_Founta_Balanced')
#model.save('Model_Result\model.Proposed_Model_Gao_Balanced')
#model.save('Model_Result\model.Proposed_Model_Roy_Balanced')
#model.save('Model_Result\model.Proposed_Model_HateXplain_Balanced')
#model.save('Model_Result\model.Proposed_Model_Davidson_Unbalanced')
#model.save('Model_Result\model.Proposed_Model_Founta_Unbalanced')
#model.save('Model_Result\model.Proposed_Model_Gao_Unbalanced')
#model.save('Model_Result\model.Proposed_Model_Roy_Unbalanced')
#model.save('Model_Result\model.Proposed_Model_HateXplain_Unbalanced')

