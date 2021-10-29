from keras import backend as K
from keras.models import load_model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes
from yad2k.models.keras_yolo import yolo_head
from app_utils import yolo_eval
from os import listdir
from os.path import isfile, join
import os

sess = K.get_session()

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)

yolo_model = load_model("model_data/yolo.h5")

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

def predict(sess, image_file):
    
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data,
                                                                                       K.learning_phase(): 0})

    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join("out", image_file), quality=90)

    return out_scores, out_boxes, out_classes

all_outputs = [predict(sess,f) for f in listdir('images/') if isfile(join('images/', f))]

print('All outputs are saved in out/ directory')