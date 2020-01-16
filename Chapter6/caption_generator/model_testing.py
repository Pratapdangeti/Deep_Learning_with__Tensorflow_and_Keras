
import pickle
import caption_model_generator
import numpy as np
from keras.preprocessing import sequence


cmg = caption_model_generator.CaptionModelGenerator()


# Function for processing captions
def process_caption(caption):
	caption_split = caption.split()
	processed_caption = caption_split[1:]
	try:
		end_of_index = processed_caption.index('<end>')
		processed_caption = processed_caption[:end_of_index]
	except:
		pass
	return " ".join([word for word in processed_caption])

# Function to get best caption post beam search
def get_best_caption(captions):
    captions.sort(key = lambda l:l[1])
    best_caption = captions[-1][0]
    return " ".join([cmg.index_to_word[index] for index in best_caption])



# Function to generate captions
# applying beam search of size 3, means always keep top 3 predictions then feed them again into themodel
def generate_captions(model, image, beam_search_size):
	start = [cmg.word_to_index['<start>']]
	captions = [[start,0.0]]
	while(len(captions[0][0]) < cmg.max_caption_len):
		captions_temporary = []
		for caption in captions:
			partial_caption_gen = sequence.pad_sequences([caption[0]], maxlen=cmg.max_caption_len, padding='post')
            # Following function to predict each word for input word and image features
			pred_next_words = model.predict([np.asarray([image]), np.asarray(partial_caption_gen)])[0]
            # Beam search is used to keep top k predictions after sorting, here is 3
			next_words = np.argsort(pred_next_words)[-beam_search_size:]
			for word in next_words:
				new_partial_caption, new_partial_caption_prob = caption[0][:], caption[1]
				new_partial_caption.append(word)
				new_partial_caption_prob+=pred_next_words[word]
				captions_temporary.append([new_partial_caption,new_partial_caption_prob])
		captions = captions_temporary
		captions.sort(key = lambda l:l[1])
		captions = captions[-beam_search_size:]
    # top k captions
	return captions

def model_testing(weight, img_name, beam_search_size = 3):
    data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter6/data/Flickr/Flickr8k_text/"
    encoded_images = pickle.load( open(data_path+"encoded_images.p","rb"))
    model = cmg.create_final_model(return_model= True)
    model.load_weights(weight)

    image = encoded_images[img_name.encode()]
    captions = generate_captions(model, image, beam_search_size)
    return process_caption(get_best_caption(captions))


if __name__ == '__main__':
    data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter6/data/Flickr/"
    weight = 'weights-epoch-49.hdf5'

    test_image_1 = '3155451946_c0862c70cb.jpg'
    print(model_testing(weight, test_image_1, beam_search_size=3))

    #test_image_2 ='1258913059_07c613f7ff.jpg'
    #print(model_testing(weight, test_image_2,beam_search_size=3))

