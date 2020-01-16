
# Method to download manually from portal directly and create dataset
import numpy as np
import os
import tarfile
import urllib.request

# local path address
# Note: Make sure you create data\CIFAR-10 folder if does not exist wherever you want
data_path = "D:\Book writing\Actual Book\Deep Learning\Codes\Chapter6\data\CIFAR-10"
file_name = os.path.join(data_path,'cifar-10-python.tar.gz')

# web address from which file needs to be downloaded incase if file was not downloaded already
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


# Check if tar file exist else download and extracts tar file
def download_extract_cifar():
    if os.path.exists(file_name):
        print("Data has already does exist")
    else:
        print("Start Downloading ... \n")
        file_path, _ = urllib.request.urlretrieve(url=data_url,filename=file_name,reporthook=None)
        print('\nExtracting... ', end='')
        tarfile.open(name=file_path, mode="r:gz").extractall(data_path)
        print('done')

# load data into train and test
import _pickle as pickle
from keras.utils import np_utils

image_size = 32
image_channels = 3
num_classes = 10
# Image is 3D which is a color image (3 channels for RGB)
image_size_flat = image_size *image_size*image_channels
# In cifar-10-batches-py folder 5 files for training with data_batch_1 to data_batch_5
# for testing file called test_batch
num_files_train = 5
images_per_file = 10000
num_images_train = num_files_train * images_per_file

# Method of loading data
def load_data(file_to_load):
    # check path join in case of issues
    file_path = os.path.join(data_path,"cifar-10-batches-py",file_to_load)
    print("loading :",file_to_load)
    with open(file_path,mode='rb') as file:
        data = pickle.load(file,encoding='bytes')
    raw_images = data[b'data']
    clsses = np.array(data[b'labels'])
    shaped_images =  raw_images.reshape([-1,image_channels, image_size,image_size])
    shaped_images = np.rollaxis(shaped_images,1,4)
    return shaped_images,clsses

# Method for loading training data
def load_training_data():
    # Initially create an array to accomodate data from all the 5 batches
    images = np.zeros(shape=[num_images_train,image_size,image_size,image_channels],dtype=np.int)
    clsses = np.zeros(shape=[num_images_train],dtype=np.int)

    start =0
    for i in range(num_files_train):
        images_batch,clsses_batch = load_data(file_to_load="data_batch_"+str(i+1))
        num_images = len(images_batch)
        end = start + num_images
        # Here update larger created array by copying with each batch
        images[start:end,:]=images_batch
        clsses[start:end] = clsses_batch
        start = end
    return images, np_utils.to_categorical(clsses,num_classes)

# loading test data separately as the entire data does come in one file only rather than batches
def load_testing_data():
    test_images,test_clsses = load_data(file_to_load="test_batch")
    return test_images, np_utils.to_categorical(test_clsses,num_classes)


#  Data loading for training and testing
download_extract_cifar()
x_train,y_train = load_training_data()
x_test,y_test = load_testing_data()


print("X train shape :",x_train.shape)
print("y train shape :",y_train.shape)
print("X test shape :",x_test.shape)
print("y test shape :",y_test.shape)





