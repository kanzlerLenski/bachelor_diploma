# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/gdrive')

pip install pymorphy2

import nltk
nltk.download('punkt')

# import all necessary models.

# word tokenizer.
from nltk import word_tokenize

# unicode character database.
import unicodedata as u

# gensim implementation of word2vec.
from gensim.models import Word2Vec

# a library that support multi-dimensional arrays, matrices and 
# high-level mathematical functions.
import numpy as np

# morphological analyzer.
import pymorphy2

# module for tagset vectorization.
from get_tag_vector import get_tag_vector

# random number generator.
import random

# module to store files in json-format.
import json

# keras functions for a model.
from keras.layers import Input, Embedding, Dense, LSTM, SpatialDropout1D, \
Activation, concatenate, Bidirectional
from keras.models import Model, load_model
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping

morph = pymorphy2.MorphAnalyzer()

# read a corpus.
text = open('/content/tales.txt', encoding='utf-8').read().lower()

# calculate statistics. 
print('total corpus:', len(text))
tokens = word_tokenize(text)
corpus_len = len(tokens)
print('total tokens:', corpus_len)

# load trained word2vec model.
w2v_model = Word2Vec.load('/content/word2vec_model')
print('total lemmas:', len(w2v_model.wv.vocab))

# a function for getting index of a word in word2vec vocabulary.
def word2idx(word):
    return w2v_model.wv.vocab[word].index

unique_tokens = sorted(list(set(tokens)))
print('unique_tokens:', len(unique_tokens))

# a dictionary of all words that can be predicited, 
# their lemmas, tagsets and grammeme vectors. 
unique_dict = {}
for token in unique_tokens:
    p = morph.parse(token)[0]
    lemma = p.normal_form
    grammeme = p.tag   
    if lemma in w2v_model.wv.vocab:
        vector = get_tag_vector(grammeme)
        unique_dict[token] = (lemma, (grammeme, vector))
        
print('unique_dict:', len(unique_dict))

# 2 dictionaries to get word be index and, the opposite, index by word.
idx2tok = dict((i, t) for i, t in enumerate(unique_dict.keys()))
tok2idx = dict((t, i) for i, t in enumerate(unique_dict.keys()))

unique_grammems = []
seen = set()
for key in unique_dict:
    if unique_dict[key][1][0] not in seen:
        seen.add(unique_dict[key][1][0])
        unique_grammems.append(unique_dict[key][1])    
print('unique_grammemes:', len(unique_grammems))

# max length of a sequence to consider for prediction 
# of the next character.
max_sent_len = 17

# break corpus into sequences of the max_sent_len length. 
sequences = []
for i in range(max_sent_len, len(tokens)):
	seq = tokens[i-max_sent_len:i]
	sequences.append(seq)	
print('total sequences:', len(sequences))

# dimension of each grammeme vector.
grammeme_vector_dimension = 59

# vectorization of lemmas, grammemes and words to be predicted by them.
x_lemmas = np.zeros((len(sequences), max_sent_len), dtype=np.int)
x_grammemes = np.zeros((len(sequences), max_sent_len, 
                        grammeme_vector_dimension), dtype=np.int)
y_word = np.zeros((len(sequences)), dtype=np.int)

for i, seq in enumerate(sequences):
    for t, token in enumerate(seq[:-1]):

        try:
            x_lemmas[i, t] = word2idx(unique_dict[token][0])
            x_grammemes[i, t] = unique_dict[token][1][1]
            y_word[i] = tok2idx[seq[-1]]
        except KeyError:
            pass

a = 194408  
for i in range(1,a):
    if a % i == 0:
        print(i)

# building of a model.
embedding_size = len(w2v_model.wv.vocab)
embeddings_dimension = 150
lemmas = Input(shape=(None,), name='lemmas')
lemmas_embedding = Embedding(embedding_size, embeddings_dimension, 
                             name='embeddings')(lemmas)
lemmas_embedding = SpatialDropout1D(.3)(lemmas_embedding)

grammeme_dense_units = (59, 39)
grammemes_input = Input(shape=(None, grammeme_vector_dimension), 
                        name='grammemes')
grammemes_layer = grammemes_input
for grammeme_dense_layer_units in grammeme_dense_units:
    grammemes_layer = Dense(grammeme_dense_layer_units, 
                            activation='relu')(grammemes_layer)

lstm_units = 128
dropout = 0.6
dense_units = 256
layer = concatenate([lemmas_embedding, grammemes_layer], name="LSTM_input")
#layer = Bidirectional(LSTM(lstm_units, dropout=dropout, recurrent_dropout=dropout,
#             return_sequences=True, name='LSTM_1'))(layer)
layer = Bidirectional(LSTM(lstm_units, dropout=dropout, recurrent_dropout=dropout,
             return_sequences=False, name='LSTM_1'))(layer)

layer = Dense(dense_units)(layer)
layer = Activation('relu')(layer)

softmax_size = len(unique_dict)
output = Dense(softmax_size, activation='softmax')(layer)

model = Model(inputs=[lemmas, grammemes_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
print(model.summary())

# a function to sample index probability. 
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# a function for generating a new tale.
def generate_next(words, diversity, num_generated):
    word_idxs = [tok2idx[w] for w in words]
    for i in range(num_generated):
        x_lemmas = np.zeros((1, len(words)))
        x_grammemes = np.zeros((1, len(words), grammeme_vector_dimension))

        for n, word in enumerate(words):
            x_lemmas[0, n] = word2idx(unique_dict[word][0])
            x_grammemes[0, n] = unique_dict[word][1][1]
        prediction = model.predict([x_lemmas, x_grammemes])
        next_idx = sample(prediction[-1], diversity) 
        word_idxs.append(next_idx)
    return ' '.join(idx2tok[idx] for idx in word_idxs)

# a function called for generation after each epoch.
def on_epoch_end(epoch, _):

	# to begin prediction random sequence is taken from a corpus.
	# that sequence is randomly chosen among all sequences 
	# of max_sent_len length containing a starting word
	# which is also chosen randomly.
    print('\nGenerating text after epoch: %s' % str(epoch + 1))
    start_index = random.choice(list(idx2tok.keys()))
    to_word = idx2tok[start_index]
    in_text = [i for i, x in enumerate(tokens) if x == to_word]
    seq = []
    for idx in in_text:
        seq.append(tokens[idx:idx+70])

    random_seq = random.choice(seq)
	
	# all words that the model 'doesn't know' are deleted from  
	# the sequence.
    words = []
    for w in random_seq:
        if len(words) < max_sent_len and w in unique_dict.keys():
            words.append(w)
	
	# generate texts with different temperatures.
    for diversity in [0.2, 0.5, 0.7, 1.0, 1.2]:
        print('----- diversity:', diversity)

        sample = generate_next(words, diversity, 100)
        print('%s... -> %s' % (' '.join(words), sample))

# save best weights.        
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                             save_best_only=True, mode='min')
lambdacall = LambdaCallback(on_epoch_end=on_epoch_end)
#earlystopping = EarlyStopping(monitor='val_acc', patience=5, 
#                               restore_best_weights=True)
callbacks_list = [checkpoint, lambdacall]#, earlystopping]

# save history.
history = model.fit([x_lemmas, x_grammemes], y_word,
          batch_size=152, 
          epochs=100,
          callbacks=callbacks_list,
          validation_split=0.1)

with open('/content/gdrive/My Drive/history.json', 'w') as f:
    json.dump(history.history, f)

# save the final model.
model.save('/content/gdrive/My Drive/model.hdf5')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_next(words, diversity, num_generated):
    word_idxs = [tok2idx[w] for w in words]
    for i in range(num_generated):
        x_lemmas = np.zeros((1, len(words)))
        x_grammemes = np.zeros((1, len(words), grammeme_vector_dimension))

        for n, word in enumerate(words):
            x_lemmas[0, n] = word2idx(unique_dict[word][0])
            x_grammemes[0, n] = unique_dict[word][1][1]
        prediction = model.predict([x_lemmas, x_grammemes])
        next_idx = sample(prediction[-1], diversity) 
        word_idxs.append(next_idx)
    return ' '.join(idx2tok[idx] for idx in word_idxs)

model = load_model('/content/model.hdf5')
start_index = random.choice(list(idx2tok.keys()))
to_word = idx2tok[start_index]
in_text = [i for i, x in enumerate(tokens) if x == to_word]
seq = []
for idx in in_text:
    seq.append(tokens[idx:idx+70])

random_seq = random.choice(seq)

words = []
for w in random_seq:
    if len(words) < max_sent_len and w in unique_dict.keys():
        words.append(w)

for diversity in [0.2, 0.5, 0.7, 1.0, 1.2]:
    print('----- diversity:', diversity)

    final = generate_next(words, diversity, 100)
    print('%s... -> %s' % (' '.join(words), final))