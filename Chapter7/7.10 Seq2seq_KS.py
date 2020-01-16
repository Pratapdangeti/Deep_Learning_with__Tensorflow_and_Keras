
data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter7/data/"

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,RepeatVector,TimeDistributed
from keras.layers.recurrent import LSTM
pd.set_option('display.max_columns',None)


# Reading the file
with open(data_path+"bot.txt", 'r') as content_file:
    botdata = content_file.read()

# Extracting question and answer pairs from text
Questions = []
Answers = []

for line in botdata.split("</pattern>"):
    if "<pattern>" in line:
        Quesn = line[line.find("<pattern>" ) +len("<pattern>"):]
        Questions.append(Quesn.lower())

for line in botdata.split("</template>"):
    if "<template>" in line:
        Ans = line[line.find("<template>" ) +len("<template>"):]
        Ans = Ans.lower()
        Answers.append(Ans.lower())


QnAdata = pd.DataFrame(np.column_stack([Questions ,Answers]) ,columns = ["Questions" ,"Answers"])
QnAdata["QnAcomb"] = QnAdata["Questions" ] +"  " +QnAdata["Answers"]

print(QnAdata.head())

# Creating Vocabulary
import nltk
import collections
counter = collections.Counter()

for i in range(len(QnAdata)):
    for word in nltk.word_tokenize(QnAdata.iloc[i][2]):
        counter[word]+=1

word2idx = {w :(i +1) for i ,(w ,_) in enumerate(counter.most_common())}
idx2word = {v :k for k ,v in word2idx.items()}

idx2word[0] = "PAD"
vocab_size = len(word2idx) +1
print ("\n\nVocabulary size:" ,vocab_size)

# Encoding and decoding function as the only langauge is english
def encode(sentence, maxlen, _vocab_size):
    indices = np.zeros((maxlen, _vocab_size))
    for i, w in enumerate(nltk.word_tokenize(sentence)):
        if i == maxlen: break
        indices[i, word2idx[w]] = 1
    return indices

def decode(indices, calc_argmax=True):
    if calc_argmax:
        indices = np.argmax(indices, axis=-1)
    return ' '.join(idx2word[x] for x in indices)


question_maxlen = 10
answer_maxlen = 20

def create_questions(_question_maxlen, _vocab_size):
    question_idx = np.zeros(shape=(len(Questions) , _question_maxlen , _vocab_size))
    for q in range(len(Questions)):
        question = encode(Questions[q], _question_maxlen, _vocab_size)
        question_idx[i] = question
    return question_idx

def create_answers(_answer_maxlen, _vocab_size):
    answer_idx = np.zeros(shape=(len(Answers) , _answer_maxlen , _vocab_size))
    for q in range(len(Answers)):
        answer = encode(Answers[q], _answer_maxlen, _vocab_size)
        answer_idx[i] = answer
    return answer_idx

quesns_train = create_questions(_question_maxlen=question_maxlen, _vocab_size=vocab_size)
answs_train = create_answers(_answer_maxlen=answer_maxlen, _vocab_size=vocab_size)


# Model architecture
model = Sequential()
model.add(LSTM(150, input_shape=(question_maxlen, vocab_size)))
model.add(RepeatVector(answer_maxlen))
model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


print (model.summary())


# Model Training
quesns_train_2 = quesns_train.astype('float32')
answs_train_2 = answs_train.astype('float32')
model.fit(quesns_train_2, answs_train_2 ,batch_size=32 ,epochs=30 ,validation_split=0.05,verbose=2)


# Model preidciton
ans_pred = model.predict(quesns_train_2[0:3])
print(decode(ans_pred[0]))
print(decode(ans_pred[1]))


print("\nCompleted")


