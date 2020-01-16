

# From deep learning keras book

import keras
from keras import layers
import numpy as np


latent_dim = 32
height = 32
width = 32
channels = 3


from keras.layers import Dense,LeakyReLU,Reshape,Conv2D,Conv2DTranspose
from keras.models import Model
from keras.layers import Input,Flatten,Dropout

# Generator network
generator_input = Input(shape=(latent_dim,))

x = Dense(128*16*16)(generator_input)
x = LeakyReLU()(x)
x = Reshape((16,16,128))(x)

x = Conv2D(256,5,padding='same')(x)
x=LeakyReLU()(x)

x = Conv2DTranspose(256,4,strides=2,padding='same')(x)
x=LeakyReLU()(x)

x = Conv2D(256,5,padding='same')(x)
x = LeakyReLU()(x)
x = Conv2D(256,5,padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(channels,7,activation='tanh',padding='same')(x)

generator = Model(generator_input,x)
generator.summary()




# Discriminator network
discriminator_input = Input(shape=(height,width,channels))
y = Conv2D(128,3)(discriminator_input)
y = LeakyReLU()(y)
y = Conv2D(128,4,strides=2)(y)
y = LeakyReLU()(y)
y = Conv2D(128,4,strides=2)(y)

y = LeakyReLU()(y)
y = Conv2D(128,4,strides=2)(y)

y = LeakyReLU()(y)
y = Flatten()(y)

y = Dropout(0.4)(y)
y = Dense(1,activation='sigmoid')(y)

discriminator = Model(discriminator_input,y)
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.001,clipvalue=1.0,decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy')


discriminator.trainable=False
gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input,gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004,clipvalue=1.0,decay=1e-8)
gan.compile(optimizer=gan_optimizer,loss='binary_crossentropy')

import os
from keras.preprocessing import image

(x_train,y_train),(_,_) = keras.datasets.cifar10.load_data()

x_train = x_train[y_train.flatten()==6]

x_train = x_train.reshape((x_train.shape[0],) + (height,width,channels)).astype('float32')/255.

iterations = 10000
batch_size = 20


if not os.path.exists('output/ks/'):
    os.makedirs('output/ks/')

save_dir = 'output/ks/'

start = 0
for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size,latent_dim))

    generated_images = generator.predict(random_latent_vectors)

    stop = start+batch_size

    real_images = x_train[start:stop]

    combined_images = np.concatenate([generated_images,real_images])

    labels = np.concatenate([np.ones((batch_size,1)),np.zeros((batch_size,1))])
    labels += 0.05*np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images,labels)

    random_latent_vectors = np.random.normal(size=(batch_size,latent_dim))
    misleading_targets = np.zeros((batch_size,1))

    a_loss = gan.train_on_batch(random_latent_vectors,misleading_targets)

    start += batch_size
    if start > len(x_train)-batch_size:
        start=0


    if step%100 == 0:
        gan.save_weights('gan.h5')


        print('discriminator loss:',d_loss)
        print('adversarial loss:',a_loss)

        img = image.array_to_img(generated_images[0]*255.,scale=False)
        img.save(os.path.join(save_dir,'generated_frog'+str(step)+'.png'))

        img = image.array_to_img(real_images[0]*255.,scale=False)
        img.save(os.path.join(save_dir,'real_frog'+str(step)+'.png'))
























