


import os
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# Folder location for input, output, helper functions and model data
data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter6/data/YOLO"

# YOLO helper functions
from data.YOLO.helper_functions.yolo_helper import read_classes_from,preprocess_image, import_anchors,create_colors,plot_boxes, scale_boxes

# Can be downloaded from yad2k library of keras implementation
# https://github.com/allanzelener/YAD2K/tree/master/yad2k/models
from data.YOLO.helper_functions.yad2k_models_keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


# Following function filters YOLO boxes by thresholding on object and class confidence
def filtering_boxes(_confidence_of_box, _list_of_boxes, _prob_of_box_class, threshold=.6):
    _scores_of_boxes = _confidence_of_box * _prob_of_box_class
    classes_of_boxes = K.argmax(_scores_of_boxes, -1)
    _class_scores_of_box = K.max(_scores_of_boxes, -1)
    filtering_mask = _class_scores_of_box > threshold
    unfiltered_scores = tf.boolean_mask(_class_scores_of_box, filtering_mask)
    _list_of_boxes = tf.boolean_mask(_list_of_boxes, filtering_mask)
    predicted_classes = tf.boolean_mask(classes_of_boxes, filtering_mask)
    return unfiltered_scores, _list_of_boxes, predicted_classes


# Function for Intersection over union (IOU)
def intersetion_over_union(_box1, _box2):
    _x1 = max(_box1[0],_box2[0])
    _y1 = max(_box1[1],_box2[1])
    _x2 = min(_box1[2],_box2[2])
    _y2 = min(_box1[3],_box2[3])
    intersection_area = (_y2-_y1) * (_x2-_x1)
    area_box1 = (_box1[3]-_box1[1]) * (_box1[2]-_box1[0])
    area_box2 = (_box2[3]-_box2[1]) * (_box2[2]-_box2[0])
    union_area = area_box1 + area_box2 - intersection_area
    iou = intersection_area / union_area
    return iou


# Function for calculating non max supression by eliminating all the boxes except the high score one
def non_max_suppression(_scores, _boxes, _classes, max_boxes = 10, iou_threshold = 0.5):
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_indices = tf.image.non_max_suppression(_boxes, _scores, max_boxes, iou_threshold)
    _scores = K.gather(_scores, nms_indices)
    _boxes = K.gather(_boxes, nms_indices)
    _classes = K.gather(_classes, nms_indices)
    return _scores, _boxes, _classes


outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
           tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
           tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
           tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))


# Following function converts the output of YOLO encoding to predicted boxes along with scores,
# box co-ordinates and classes
def yolo_eval(_outputs,img_shape = (720.,1280.),max_boxes=10,score_threshold=.6,iou_threshold=.5):
    box_confidence, box_xy, box_wh, box_class_probs = _outputs
    _boxes = yolo_boxes_to_corners(box_xy, box_wh)
    _scores, _boxes, _classes = filtering_boxes(box_confidence,_boxes,box_class_probs,threshold = score_threshold)
    _boxes = scale_boxes(_boxes, img_shape)
    _scores,_boxes,_classes = non_max_suppression(_scores, _boxes, _classes, max_boxes, iou_threshold)
    return _scores,_boxes,_classes


scores, boxes, classes = yolo_eval(outputs)

with tf.Session() as test_b:
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))


sess = K.get_session()
class_names = read_classes_from(data_path + "/model_data/coco_classes.txt")
anchors = import_anchors(data_path + "/model_data/yolo_anchors.txt")
yolo_model = load_model(data_path+"/model_data/yolo.h5")


outputs = yolo_head(yolo_model.output, anchors, len(class_names))

# Following function use for predict the boxes on image file.
def predict(_sess, image_file):
    _image, _image_data = preprocess_image(data_path+"/input_image/" + image_file, model_image_size = (608, 608))
    _out_scores, _out_boxes, _out_classes = _sess.run([scores, boxes, classes], feed_dict={yolo_model.input: _image_data, K.learning_phase(): 0})
    print('There are {} boxes found in {}'.format(len(_out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    _colors = create_colors(class_names)
    # Draw bounding boxes on the image file
    plot_boxes(_image, _out_scores, _out_boxes, _out_classes, class_names, _colors)
    # Save the predicted bounding box on the image
    _image.save(os.path.join(data_path+"/output_image/",image_file), quality=90)
    # Display the results in the notebook
    _output_image = scipy.misc.imread(os.path.join(data_path+"/output_image/",image_file))
    plt.figure(figsize=(12,12))
    imshow(_output_image)
    return _out_scores, _out_boxes, _out_classes


img = plt.imread(data_path+"/input_image/test.jpg")
image_shape = float(img.shape[0]), float(img.shape[1])
scores, boxes, classes = yolo_eval(outputs, image_shape)
out_scores, out_boxes, out_classes = predict(sess,"test.jpg")


print("Finished !")

