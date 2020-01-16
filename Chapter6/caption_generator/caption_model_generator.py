#https://github.com/anuragmishracse/caption_generator


import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import LSTM, Embedding, TimeDistributed, \
    Dense, RepeatVector, add, Input,Dropout
from keras.preprocessing import image, sequence
import pickle


data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter6/data/Flickr/"


EMBEDDING_DIM = 128
class CaptionModelGenerator:
    # Variables defined
    def __init__(self):
        self.total_samples = None
        self.max_caption_len = None
        self.vocabulary_size = None
        self.word_to_index = None
        self.index_to_word = None
        self.encoded_images = pickle.load(open( data_path +"Flickr8k_text/encoded_images.p","rb"))
        self.variable_initializer()

    # Variables initilization with appropriate values
    def variable_initializer(self):
        df = pd.read_csv(data_path +'Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        num_samples = df.shape[0]
        iter = df.iterrows()
        captions_list = []
        for i in range(num_samples):
            x = iter.__next__()
            captions_list.append(x[1][1])

        self.total_samples=0
        for text in captions_list:
            self.total_samples+=len(text.split())-1
        print ("Total number of data samples : "+str(self.total_samples))
        
        words = [txt.split() for txt in captions_list]
        unique = []
        for word in words:
            unique.extend(word)
        # Converting every word into respective number by creating unique words in set
        unique = list(set(unique))
        self.vocabulary_size = len(unique)
        self.word_to_index = {}
        self.index_to_word = {}
        for i, word in enumerate(unique):
            self.word_to_index[word]=i
            self.index_to_word[i]=word
        # Maximum length to be allowed for caption determined by caption with maximum number of words
        max_len = 0
        for caption in captions_list:
            if len(caption.split()) > max_len :
                max_len = len(caption.split())
        self.max_caption_len = max_len
        print ("Vocabulary size: " + str(self.vocabulary_size))
        print ("Maximum caption length: " + str(self.max_caption_len))
        print ("Variables initialization completed!")


    # In the follwing function data generated using generator functions
    def data_generator(self, batch_size = 32):
        partial_caps = []
        next_words = []
        images = []

        gen_count = 0
        df = pd.read_csv(data_path+'Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iters = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = iters.__next__()
            caps.append(x[1][1])
            imgs.append(x[1][0])

        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter+=1
                current_image = self.encoded_images[imgs[image_counter].encode()]
                for i in range(len(text.split())-1):
                    total_count+=1
                    partial = [self.word_to_index[txt] for txt in text.split()[:i + 1]]
                    partial_caps.append(partial)
                    next = np.zeros(self.vocabulary_size)
                    next[self.word_to_index[text.split()[i + 1]]] = 1
                    next_words.append(next)
                    images.append(current_image)

                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_caption_len, padding='post')
                        total_count = 0
                        gen_count+=1
                        #print ("yielding count: "+str(gen_count))
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []

    # Load images as numpy array
    def load_image(self, path):
        img = image.load_img(path, target_size=(224,224))
        x = image.img_to_array(img)
        return np.asarray(x)

    # Creating the final deep learning model for fitting features with caption vocabulary
    def create_final_model(self, return_model = False):

        # Image features input
        image_input = Input(shape=(4096,))
        image_layer_1 = Dropout(0.5)(image_input)
        image_layer_2 = Dense(EMBEDDING_DIM, activation='relu')(image_layer_1)
        image_layer_3 = RepeatVector(self.max_caption_len)(image_layer_2)

        # sequence model - input caption sequence
        caption_input = Input(shape=(self.max_caption_len,))
        caption_layer_1 = Embedding(self.vocabulary_size, 256, mask_zero=True)(caption_input)
        caption_layer_2 = Dropout(0.5)(caption_layer_1)
        caption_layer_3 = LSTM(256, return_sequences=True)(caption_layer_2)
        caption_layer_4 = TimeDistributed(Dense(EMBEDDING_DIM))(caption_layer_3)

        # decoder model - output caption sequence
        merge_layer_1 = add([image_layer_3, caption_layer_4])
        merge_layer_2 = LSTM(1000, return_sequences=False)(merge_layer_1)
        output_layer = Dense(self.vocabulary_size, activation='softmax')(merge_layer_2)

        model = Model(inputs=[image_input, caption_input], outputs=output_layer)
        print ("Model created!")

        if return_model==True:
            return model

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    # Function for obtaining word from index
    def get_word(self,index):
        return self.index_to_word[index]
