
import numpy as np
import pickle
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# global counter
counter = 0

# Change location where dataset is saved
data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter6/data/Flickr/"

# Train images
file_train_images = open(data_path + 'Flickr8k_text/Flickr_8k.trainImages.txt', 'rb')
train_imgs = file_train_images.read().splitlines()
file_train_images.close()
print(len(train_imgs))

# Test images
file_test_images = open(data_path + 'Flickr8k_text/Flickr_8k.testImages.txt', 'rb')
test_imgs = file_test_images.read().splitlines()
file_test_images.close()
print(len(test_imgs))

# Creation of new dataset for image and its respective captions
file_train_dataset = open(data_path + 'Flickr8k_text/flickr_8k_train_dataset.txt', 'wb')
file_train_dataset.write(b"image_id\tcaptions\n")

file_test_dataset = open(data_path + 'Flickr8k_text/flickr_8k_test_dataset.txt', 'wb')
file_test_dataset.write(b"image_id\tcaptions\n")

file_captions = open(data_path + 'Flickr8k_text/Flickr8k.token.txt', 'rb')
captions_list = file_captions.read().strip().split(b'\n')

data = {}
for row_c in captions_list:
    row_c = row_c.split(b"\t")
    row_c[0] = row_c[0][:len(row_c[0]) - 2]
    try:
        data[row_c[0]].append(row_c[1])
    except:
        data[row_c[0]] = [row_c[1]]
file_captions.close()

# Importing of Pre-trained model for feature extraction from image - transfer learning
encoded_images = {}
vgg_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
encoding_model = Model(input=vgg_model.input, output=vgg_model.get_layer('fc2').output)

# Function for loading image into array
def load_image_to_array(path):
    # target size argument in below function outputs into the same size irrespective of input size
    img_ld = image.load_img(path, target_size=(224,224))
    img_arr = image.img_to_array(img_ld)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    return np.asarray(img_arr)

# Function for encoding/feature extraction from image
def get_encoding_img(model, img):
	global counter
	counter += 1
	image = load_image_to_array(str(data_path + 'Flicker8k_Dataset/' + img.decode("utf-8")))
	pred_features = model.predict(image)
	pred_features = np.reshape(pred_features, pred_features.shape[1])
	print ("Encoding image: "+str(counter))
	print (pred_features.shape)
	return pred_features

# Application of feature extraction function on train images
count_train = 0
for img in train_imgs:
    encoded_images[img] = get_encoding_img(encoding_model, img)
    for capt in data[img]:
        caption = b"<start> " + capt + b" <end>"
        file_train_dataset.write(img + b"\t" + caption + b"\n")
        file_train_dataset.flush()
        count_train += 1
file_train_dataset.close()

# Application of feature extraction function on test images
count_test = 0
for img in test_imgs:
    encoded_images[img] = get_encoding_img(encoding_model, img)
    for capt in data[img]:
        caption = b"<start> " + capt + b" <end>"
        file_test_dataset.write(img + b"\t" + caption + b"\n")
        file_test_dataset.flush()
        count_test += 1
file_test_dataset.close()

# Saving all encoded train and test data into pickle file
with open(data_path +"Flickr8k_text/encoded_images.p", "wb") as pickle_f:
    pickle.dump(encoded_images, pickle_f)


print("Code finished!")

