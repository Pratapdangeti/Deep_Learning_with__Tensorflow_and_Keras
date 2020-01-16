

import tensorflow as tf
# Eager execution needs to be enabled to run batch size iteration
tf.enable_eager_execution()
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import time
from itertools import islice


data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter7/data/"
# Input file download - english to spanish translation file
path_to_file = data_path+"spa.txt"


# Number of example pairs to limit
num_datapoints = 25000

lines = open(path_to_file, encoding='UTF-8').read().strip().split('\n')

# Converts from unicode to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# pre-processing of input text
def preprocessing(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating the space between a word and punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing all the characters with space, except "a-z","A-Z",".","!","?",","
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    # adding start and end keyword to all words to represent start and end
    w = '<start> ' + w + ' <end>'
    return w

# word pairs in list of lists
word_pairs = [[preprocessing(w) for w in l.split('\t')] for l in lines[:num_datapoints]]


# Class created to creates a separate objects for 2 languages
class Lang_Index_Cls():
    def __init__(self,lang_list):
        self.lang_list = lang_list
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_indexes()

    def create_indexes(self):
        for sent in self.lang_list:
            self.vocab.update(sent.split(' '))
        # sorting vocab
        self.vocab = sorted(self.vocab)
        # create 0 pad as starting of vocab and remaining words starts with 1
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1
        # Just creating reverse mapping of word2idx as idx2word
        for word, index in self.word2idx.items():
            self.idx2word[index] = word


# index language using the class defined above
# Spanish sentences considered input language
input_language = Lang_Index_Cls(spn for eng, spn in word_pairs)
# English sentences considered output language
target_language = Lang_Index_Cls(eng for eng, spn in word_pairs)

#Return first n items of the iterable as a list"
def take(n, iterable):
    return list(islice(iterable, n))

print("\nSample 10 words from spanish vocab: \n", take(10, input_language.vocab))
print("\nSample 10 key,value spanish pairs from word2idx: \n",
      {k: input_language.word2idx[k] for k in list(input_language.word2idx)[:10]})
print("\nSample 10 key,value spanish pairs from idx2word: \n",
      {k: input_language.idx2word[k] for k in list(input_language.idx2word)[:10]})

print("\nSample 10 words from english vocab: \n",take(10,target_language.vocab))
print("\nSample 10 key,value english pairs from word2idx: \n",
      {k:target_language.word2idx[k] for k in list(target_language.word2idx)[:10]})
print("\nSample 10 key,value english pairs from idx2word: \n",
      {k:target_language.idx2word[k] for k in list(target_language.idx2word)[:10]})

# Vectorize the input and target languages
# Spanish sentences
input_tensor = [[input_language.word2idx[s] for s in spn.split(' ')] for eng, spn in word_pairs]

# English sentences
target_tensor = [[target_language.word2idx[s] for s in eng.split(' ')] for eng, spn in word_pairs]


def maximum_length(_tensor):
    return max(len(t) for t in _tensor)

# Calculate max_length of input and output tensor

max_length_input, max_length_target = maximum_length(input_tensor), maximum_length(target_tensor)

print("\nMaximum length of input tensor :",max_length_input)
print("\nMaximum length of target tensor :",max_length_target)

# Padding the input and output tensor to the maximum length
input_tensor_2 = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                maxlen=max_length_input,padding='post')

target_tensor_2 = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                maxlen=max_length_target,padding='post')



# Creating training and validation sets using an 90-10 split
input_tensor_train, input_tensor_test, target_tensor_train, target_tensor_test = train_test_split(input_tensor_2, target_tensor_2, test_size=0.1)

# Train shape
print("\nTrain input shape: ",len(input_tensor_train),"Train target shape: ",len(target_tensor_train))
# Test shape
print("\nTest input shape: ",len(input_tensor_test),"Test target shape: ",len(target_tensor_test))


# Parameters
Batch_size = 64
embedding_dim = 256
hidden_units = 1024


vocab_input_size = len(input_language.word2idx)
vocab_target_size = len(target_language.word2idx)
print("input vocab size :", vocab_input_size)
print("target vocab size :",vocab_target_size)

Buffer_size = len(input_tensor_train)
Num_batches = Buffer_size//Batch_size

# Make sure to enable eager execution to run the following batch slices
# Training dataset shuffled and created into batches
train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(Buffer_size)
train_dataset = train_dataset.batch(Batch_size, drop_remainder=True)


# Following function applied CuDNNGRU if GPU does exist on your computer
# CuDNNGRU can get about 3x speed compare with simple GRU function
def custom_gru(_units):
  if tf.test.is_gpu_available():
    return tf.keras.layers.CuDNNGRU(_units,return_sequences=True,
                        return_state=True,recurrent_initializer='glorot_uniform')
  else:
    return tf.keras.layers.GRU(_units,return_sequences=True,
                        return_state=True,recurrent_activation='sigmoid',
                        recurrent_initializer='glorot_uniform')


# Encoder class definition
class Encoder(tf.keras.Model):
    def __init__(self,_vocab_size,_embedding_dim,_encode_units,_batch_size):
        # Initializing super class variables
        super(Encoder, self).__init__()
        self._encode_units = _encode_units
        self._batch_size = _batch_size
        self._embedding = tf.keras.layers.Embedding(_vocab_size,_embedding_dim)
        self.custom_gru = custom_gru(self._encode_units)

    def call(self, _x,_hidden):
        _x = self._embedding(_x)
        _output,_state = self.custom_gru(_x,initial_state=_hidden)
        return _output,_state

    def initialize_hidden_state(self):
        return tf.zeros((self._batch_size,self._encode_units))

# Decoder class definition
class Decoder(tf.keras.Model):
    def __init__(self,_vocab_size,_embedding_dim,_decode_units,_batch_size):
        super(Decoder,self).__init__()
        self._decode_units = _decode_units
        self._batch_size = _batch_size
        self._embedding = tf.keras.layers.Embedding(_vocab_size,_embedding_dim)
        self.custom_gru = custom_gru(self._decode_units)
        self.fc = tf.keras.layers.Dense(_vocab_size)

        # Weights used for attention network
        self.W1=tf.keras.layers.Dense(self._decode_units)
        self.W2 = tf.keras.layers.Dense(self._decode_units)
        self.V =tf.keras.layers.Dense(1)

    def call(self, _x, _hidden, _encode_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        hidden_with_time_axis = tf.expand_dims(_hidden, 1)
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(_encode_output) + self.W2(hidden_with_time_axis)))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * _encode_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        _x = self._embedding(_x)
       # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        _x = tf.concat([tf.expand_dims(context_vector, 1), _x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.custom_gru(_x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size * 1, vocab)
        _x = self.fc(output)
        return _x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self._batch_size,self._decode_units))

# Encoder and Decoder object creation
encoder = Encoder(vocab_input_size, embedding_dim,hidden_units,Batch_size)
decoder = Decoder(vocab_target_size, embedding_dim,hidden_units,Batch_size)


# Defining optimization and loss
optimizer = tf.train.AdamOptimizer()

def loss_function(real, pred):
  mask = 1 - np.equal(real, 0)
  loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
  return tf.reduce_mean(loss_)

# Training the data
Training_epochs = 10

for epoch in range(Training_epochs):
    # for printing time required for each epoch
    start = time.time()
    hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (_input, _target)) in enumerate(train_dataset):
        loss = 0
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(_input, hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([target_language.word2idx['<start>']] * Batch_size, 1)

            # feeding the target as the next input
            for t in range(1, _target.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_function(_target[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(_target[:, t], 1)

        batch_loss = (loss / int(_target.shape[1]))
        total_loss += batch_loss
        variables = encoder.variables + decoder.variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,batch,
                                                         batch_loss.numpy()))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / Num_batches))
    print('Time taken for 1 epoch {} sec\n'.format(round(time.time() - start,3)))


def evaluate(sentence, _encoder, _decoder, _inp_lang, _targ_lang, _max_length_inp, _max_length_targ):
    sentence = preprocessing(sentence)
    inputs = [_inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=_max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''

    _hidden = [tf.zeros((1, hidden_units))]
    enc_out, _enc_hidden = encoder(inputs, _hidden)
    _dec_hidden = _enc_hidden
    _dec_input = tf.expand_dims([_targ_lang.word2idx['<start>']], 0)

    for _t in range(_max_length_targ):
        _predictions, _dec_hidden, _ = decoder(_dec_input, _dec_hidden, enc_out)
        predicted_id = tf.argmax(_predictions[0]).numpy()
        result += _targ_lang.idx2word[predicted_id] + ' '
        if _targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence
        # the predicted ID is fed back into the model
        _dec_input = tf.expand_dims([predicted_id], 0)
    return result, sentence

# Following function translates the given input words
def translate(sentence,  _encoder, _decoder, _inp_lang, _targ_lang, _max_length_inp, _max_length_targ):
    result, sentence = evaluate(sentence,_encoder, _decoder, _inp_lang, _targ_lang, _max_length_inp, _max_length_targ)
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))

# Print conversion of spanish words into english
translate(u'hace mucho frio aqui.', encoder, decoder, input_language, target_language, max_length_input, max_length_target)
print("Google translation converted sentence  :","Its really cold here.")
translate(u'esta es mi vida.', encoder, decoder, input_language, target_language, max_length_input, max_length_target)
print("Google translation converted sentence  :","this is my life.")

print("\nCompleted")