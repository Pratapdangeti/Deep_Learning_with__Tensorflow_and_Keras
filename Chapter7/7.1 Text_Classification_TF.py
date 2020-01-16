



import numpy as np
import pickle
from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from nltk import pos_tag
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score,confusion_matrix


newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

x_train = newsgroups_train.data
x_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

print("List of all 20 categories:")
print(newsgroups_train.target_names)
print("\n")
print("Sample Email:")
print(x_train[0])
print("Sample Target Category:")
print(y_train[0])
print(newsgroups_train.target_names[y_train[0]])


data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter7/data/"


def preprocessing(text):
    # removing punctuations
    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())
    # Tokenizing sentences
    tokens = [word for sent in nltk.sent_tokenize(text2) for word in
              nltk.word_tokenize(sent)]
    # Converting to lower case
    tokens = [word.lower() for word in tokens]
    # stopword removal
    stopwds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwds]
    # ignore if length < 3
    tokens = [word for word in tokens if len(word)>=3]
    # Stemming
    stemmer = PorterStemmer()
    try:
        tokens = [stemmer.stem(word) for word in tokens]
    except:
        tokens = tokens
    # POS tagging for lemmatization
    tagged_corpus = pos_tag(tokens)
    Noun_tags = ['NN' ,'NNP' ,'NNPS' ,'NNS']
    Verb_tags = ['VB' ,'VBD' ,'VBG' ,'VBN' ,'VBP' ,'VBZ']
    # Custom lemmatizer
    lemmatizer = WordNetLemmatizer()
    def custom_lemmatizer(token ,tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token ,'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token ,'v')
        else:
            return lemmatizer.lemmatize(token ,'n')
    # Join back words for output
    pre_proc_text =  " ".join([custom_lemmatizer(token ,tag) for token ,tag in tagged_corpus])
    return pre_proc_text

# Uncomment and Run the following code only once, afterwards use pickle for every re-run
# for further tuning the model
"""
x_train_preprocessed  = []
for i in x_train:
    x_train_preprocessed.append(preprocessing(i))

x_test_preprocessed = []
for i in x_test:
    x_test_preprocessed.append(preprocessing(i))

# Saving all encoded train and test data into pickle file
with open(data_path +"x_train_preprocessed.p", "wb") as pickle_f:
    pickle.dump(x_train_preprocessed, pickle_f)
    
with open(data_path +"x_test_preprocessed.p", "wb") as pickle_f:
    pickle.dump(x_test_preprocessed, pickle_f)
"""


x_train_preprocessed = pickle.load(open(data_path +"x_train_preprocessed.p","rb"))
x_test_preprocessed = pickle.load(open(data_path +"x_test_preprocessed.p","rb"))


# building TFIDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2),  stop_words='english',
                             max_features= 10000 ,strip_accents='unicode',  norm='l2')

# Parameters for neural networks
input_dim = 10000
layer_1_neurons = 1000
layer_2_neurons = 500
layer_3_neurons = 50
nb_classes = 20
learning_rate = 0.1

x_train_2 = vectorizer.fit_transform(x_train_preprocessed).todense()
x_test_2 = vectorizer.transform(x_test_preprocessed).todense()

# Creating vectors for Y variables
ytr_ = np.zeros((np.shape(y_train)[0],nb_classes))
ytr_[np.arange(np.shape(y_train)[0]),y_train]=1
yts_ = np.zeros((np.shape(y_test)[0],nb_classes))
yts_[np.arange(np.shape(y_test)[0]),y_test]=1
y_train_2 = ytr_
y_test_2 = yts_

def Text_Classification_TF(_x_input):
    W1 = tf.Variable(tf.random_uniform([input_dim, layer_1_neurons]), name="W1")
    W2 = tf.Variable(tf.random_uniform([layer_1_neurons, layer_2_neurons]), name="W2")
    W3 = tf.Variable(tf.random_uniform([layer_2_neurons, layer_3_neurons]), name="W3")
    WO = tf.Variable(tf.random_uniform([layer_3_neurons, nb_classes]), name="WO")

    b1=tf.Variable(tf.zeros([layer_1_neurons]),name="b1")
    b2=tf.Variable(tf.zeros([layer_2_neurons]),name="b2")
    b3 = tf.Variable(tf.zeros([layer_3_neurons]), name="b3")
    bo=tf.Variable(tf.zeros([nb_classes]),name="bo")

    # Layer 1
    layer_1 = tf.add(tf.matmul(_x_input,W1),b1)
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1,keep_prob=0.5)

    # Layer 2
    layer_2 = tf.add(tf.matmul(layer_1,W2),b2)
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob=0.5)

    # Layer 3
    layer_3 = tf.add(tf.matmul(layer_2,W3),b3)
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob=0.5)

   # Output Layer
    output_layer = tf.add(tf.matmul(layer_3,WO),bo)
    return output_layer


# Code starts here
xs = tf.placeholder(tf.float32,[None,input_dim],name="Input_data")
ys = tf.placeholder(tf.float32,[None,nb_classes],name="output_data")

# Construct model
output = Text_Classification_TF(xs)


# Define loss and output
cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,labels=ys))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_op)


training_epochs = 2
batch_size = 64

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("/tmp/tf_textlogs",sess.graph)

    for epoch in range(training_epochs):

        batch_count = int(x_train_2.shape[0]/batch_size)
        for i in range(batch_count):
            batch_x = x_train_2[(i*batch_size): ((i+1)*batch_size),:]
            batch_y = y_train_2[(i*batch_size): ((i+1)*batch_size),:]

            trc,trs = sess.run([train_op,cost_op],feed_dict={xs:batch_x, ys:batch_y})

            print("Epoch :",epoch,"batch :",i," Train Cost :",sess.run(cost_op,feed_dict={xs:x_train_2, ys:y_train_2}),
                  "Test Cost :", sess.run(cost_op, feed_dict={xs: x_test_2, ys: y_test_2}))

        trcost = sess.run(cost_op,feed_dict={xs:x_train_2, ys:y_train_2})
        tstcost = sess.run(cost_op,feed_dict={xs:x_test_2, ys:y_test_2})
        # Writing loss values to tensorboard at each epoch
        tr_summary = tf.Summary(value=[tf.Summary.Value(tag="train loss", simple_value=trcost)])
        # +1 added to epoch only to represent the step
        writer.add_summary(tr_summary, global_step=epoch)
        tst_summary = tf.Summary(value=[tf.Summary.Value(tag="test loss", simple_value=tstcost)])
        writer.add_summary(tst_summary, global_step=epoch)
    writer.close()
    print("Optimization Finished!")


    act_amax = tf.argmax(ys,1)
    pred_amax = tf.argmax(output, 1)


    print("Multi Classification Train Confusion matrix :\n",confusion_matrix(act_amax.eval({ys:y_train_2}), pred_amax.eval({xs:x_train_2}) ))
    print("Multi Classification Test Confusion matrix :\n",confusion_matrix(act_amax.eval({ys:y_test_2}), pred_amax.eval({xs:x_test_2}) ))

    print("Multi Classification Train Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_train_2}), pred_amax.eval({xs:x_train_2})),4))
    print("Multi Classification Test Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_test_2}), pred_amax.eval({xs:x_test_2})),4))

    sess.close()









