



import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential,Model
from keras.layers import Dense,LeakyReLU,Dropout,Input
import os
from keras.preprocessing import image

# Importing MNIST data
(x_train,y_train),(_,_) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],784)

# Select the number you would like to train upon
# here required digit has been selected
# Note: By training on all the digits will create noise
x_train = x_train[y_train.flatten()==2]


def adam_optimizer():
    return Adam(lr=0.001)


# Generative model
def generator():
    gen_model = Sequential()
    gen_model.add(Dense(units=128, input_dim=100))
    gen_model.add(LeakyReLU(0.2))
    gen_model.add(Dense(units=784, activation='sigmoid'))
    gen_model.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return gen_model

# Discriminator model
def discriminator():
    disc_model = Sequential()
    disc_model.add(Dense(units=128,input_dim=784))
    disc_model.add(LeakyReLU(0.2))
    disc_model.add(Dropout(0.3))
    disc_model.add(Dense(units=1, activation='sigmoid'))
    disc_model.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return disc_model

# GAN model
def create_gan(_discriminator, _generator):
    _discriminator.trainable=False
    gan_input = Input(shape=(100,))
    gan_output= _discriminator(_generator(gan_input))
    _gan= Model(inputs=gan_input, outputs=gan_output)
    _gan.compile(loss='binary_crossentropy', optimizer='adam')
    return _gan

# Model initiation
discriminator_model = discriminator()
generator_model = generator()
gan_model = create_gan(discriminator_model,generator_model)

# Create folder path if does not exist
if not os.path.exists('output/ks/'):
    os.makedirs('output/ks/')
save_dir = 'output/ks/'

# Model parameters
iterations = 200
batch_size = 20
latent_dim = 100

start = 0

for step in range(iterations):
    # Creating random noise data with dimensions as latent dim
    random_latent_vectors = np.random.normal(size=(batch_size,latent_dim))
    generated_images = generator_model.predict(random_latent_vectors)

    stop = start+batch_size
    real_images = x_train[start:stop]

    # Combining real images with generated images
    combined_images = np.concatenate([generated_images,real_images])
    labels = np.concatenate([np.ones((batch_size,1)),np.zeros((batch_size,1))])

    labels += 0.05*np.random.random(labels.shape)
    # calculating discriminator loss
    d_loss = discriminator_model.train_on_batch(combined_images,labels)

    # Training gan for generative loss
    random_latent_vectors = np.random.normal(size=(batch_size,latent_dim))
    misleading_targets = np.zeros((batch_size,1))
    g_loss = gan_model.train_on_batch(random_latent_vectors, misleading_targets)

    # Increase batch size every time
    start += batch_size
    if start > len(x_train)-batch_size:
        start=0

    # Print metrics at each increment of 10 steps
    if step%10 == 0:
        print('epoch: ', step,'discriminator loss: ', d_loss,'gen loss: ', g_loss)

        gen_img = generated_images[0].reshape(28,28,1)
        gen_img_2 = image.array_to_img(gen_img*255.,scale=False)
        gen_img_2.save(os.path.join(save_dir,'generated_number_'+str(step)+'.png'))

        real_img = real_images[0].reshape(28,28,1)
        real_img_2 = image.array_to_img(real_img*255.,scale=False)
        real_img_2.save(os.path.join(save_dir,'real_number_'+str(step)+'.png'))
        

print("Completed!")

