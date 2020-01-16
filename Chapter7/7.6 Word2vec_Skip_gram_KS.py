


from sklearn.model_selection import train_test_split
import nltk
import numpy as np
from nltk.corpus import stopwords
import string

from keras.layers import Dense,Activation,Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter7/data/"

def preprocessing(text):
    # removing punctuations
    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text.decode()]).split())
    # Tokenizing sentences
    tokens = [word for sent in nltk.sent_tokenize(text2) for word in nltk.word_tokenize(sent)]
    # Convert to lower case
    tokens = [word.lower() for word in tokens]
    # removal of stopwords
    stopwds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwds]
    # removing length less than 3
    tokens = [word for word in tokens if len(word)>=3]
    pre_proc_text = " ".join(tokens)
    return pre_proc_text


lines = []
fin = open(data_path+"Smart_Bomb_with_Language_parser.txt", "rb")
#fin = open(data_path+"alice_in_wonderland.txt", "rb")

for line in fin:
    line = line.strip().decode("ascii", "ignore").encode("utf-8")

    if len(line) == 0:
        continue
    lines.append(preprocessing(line))
fin.close()


import collections
counter = collections.Counter()

for line in lines:
    for word in nltk.word_tokenize(line):
        counter[word.lower()]+=1

word2idx = {w:(i+1) for i,(w,_) in enumerate(counter.most_common())}
idx2word = {v:k for k,v in word2idx.items()}


# Window size to control the nearest words vicinity
window_size = 2

x_append = []
y_append = []

for line in lines:
    embedding = [word2idx[w.lower()] for w in nltk.word_tokenize(line)]

    for word_index, word in enumerate(embedding):
        for nb_word in embedding[max(word_index - window_size, 0): min(word_index + window_size, len(embedding)) + 1]:
            if nb_word != word:
                # for CBOW we need to predict word using nearby words by averaging
                # for skip-gram words get swapped between x and y without any averaging
                x_append.append(word)
                y_append.append(nb_word)

vocab_size = len(word2idx)+1

X = to_categorical(x_append, num_classes=vocab_size)
Y = to_categorical(y_append, num_classes=vocab_size)


# Training and testing classification
x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=0.05,random_state=42)

# Printing dimensions
print("X_train: ",x_train.shape,"y_train: ",y_train.shape,"X_test: ",x_test.shape,"y_test: ",y_test.shape)

Embedding_dim = 5

model = Sequential()
model.add(Dense(Embedding_dim, input_shape=(vocab_size,),name="fc1"))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(vocab_size,name="fc2"))
model.add(Activation('softmax'))
adam_opt = Adam(lr=0.01)
# Model compilation
model.compile(loss='categorical_crossentropy', optimizer=adam_opt)


# Parameters of the model
learning_rate = 0.01
training_epochs = 100
batch_size = 64

model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=batch_size,epochs=training_epochs,verbose=2)


# Extracting the weights from layer 1 post training the model
W1, b1 = model.get_layer('fc1').get_weights()

# Finding intermediate/hidden layer vectors for predictions post training
vectors = W1+b1


# Finding nearest vectors and words
def euclidean_dist(vector1, vector2):
    return np.sqrt(np.sum((vector1-vector2)**2))

def find_closest(_word_index, _vectors):
    list_tuples = []
    query_vector = _vectors[_word_index]
    for index, vector in enumerate(_vectors):
        if not np.array_equal(vector, query_vector):
            list_tuples.append((euclidean_dist(vector, query_vector),index))
    sorted_list = sorted(list_tuples)
    return sorted_list


# Print top nearest words
# note: Use only small letters

nearest_word_of = 'wolf'
closest_tuples = find_closest(word2idx[nearest_word_of], vectors)
top_k = 5
print("\nTop",top_k,"similar words for: ",nearest_word_of)
for i in range(top_k):
    print("distance :",closest_tuples[i][0],", word: ",idx2word[closest_tuples[i][1]])




# tsne (manifold learning) vector visualization
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing

tsne_model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
tsne_vectors = tsne_model.fit_transform(vectors)

normalizer = preprocessing.Normalizer()
tsne_vectors =  normalizer.fit_transform(tsne_vectors, 'l1')

# List of all words
words = [k for k in word2idx]
fig, ax = plt.subplots(figsize=(8,8))

plt.xlim(-1.0,1.2)
plt.ylim(-1.5,1.2)

for word_i in range(150,200):
    print(word_i,words[word_i], tsne_vectors[word2idx[words[word_i]]][1])
    ax.annotate(words[word_i],(tsne_vectors[word2idx[words[word_i]]][0],tsne_vectors[word2idx[words[word_i]]][1]),size=8)
plt.show()



print("completed")


