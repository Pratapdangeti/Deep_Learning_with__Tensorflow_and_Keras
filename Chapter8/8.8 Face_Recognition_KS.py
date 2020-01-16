

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model,Sequential
from keras.layers import Convolution2D,ZeroPadding2D,MaxPooling2D,Flatten,Dropout,Activation


data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter8/"

# Building Model
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))

model.add(Convolution2D(4096,(7,7),activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096,(1,1),activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622,(1,1)))
model.add(Flatten())
model.add(Activation('softmax'))

#https://drive.google.com/drive/folders/1Gf_KP4yjwmKXStmFowZ0WPntx50lsX5I
# Loading weights
model.load_weights(data_path+'data/vgg_face_weights.h5')

print(model.summary())

# Pre-process convert any type of pixel size into fixed size of 224 x 224
def preprocess_image(_path):
    img = load_img(_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    return img

# Calculates cosine similarity between 2 vectors
def calculate_CosineSimilarity(_source_vector, _test_vector):
    a = np.matmul(np.transpose(_source_vector), _test_vector)
    b = np.sqrt(np.sum(np.multiply(_source_vector, _source_vector)))
    c = np.sqrt(np.sum(np.multiply(_test_vector, _test_vector)))
    return a / (b * c)

# Caclculates euclidean distances between 2 vectors
def calculate_EuclideanDistance(_source_vector, _test_vector):
    _distance = _source_vector - _test_vector
    euclidean_distance = np.sqrt(np.sum(np.multiply(_distance,_distance)))
    return euclidean_distance

# Extracting the input and last second layer from the pre-built model
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

print(vgg_face_descriptor.summary())

# Calculate similarity between 2 images
def check_Face_Similarity(img1, img2, _epsilon):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(data_path+'data/'+img1))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(data_path+'data/'+img2))[0,:]

    cosine_similarity = calculate_CosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = calculate_EuclideanDistance(img1_representation, img2_representation)

    print('-----------------------------')
    print("Cosine similarity: ",cosine_similarity)
    print("Euclidean distance: ",euclidean_distance)

    if cosine_similarity > _epsilon:
        print("Faces are matching!")
    else:
        print("Faces are not matching!")

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(image.load_img(data_path+'data/'+img1))
    plt.xticks([])
    plt.yticks([])
    fig.add_subplot(1, 2, 2)
    plt.imshow(image.load_img(data_path+'data/'+img2))
    plt.xticks([])
    plt.yticks([])
    plt.show(block=True)


epsilon = 0.60
check_Face_Similarity("Image_1.png", "Image_2.jpg", _epsilon=epsilon)
check_Face_Similarity("Image_1.png","Image_3.png",_epsilon=epsilon)


print("Completed!")

