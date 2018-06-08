from keras import backend as K
import tensorflow as tf
from keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
import shutil
import os



import argparse
import os
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def _main_(args):
 
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    K.set_learning_phase(0)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend        = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    print(weights_path)
    yolo.load_weights(weights_path)

    export_base = 'tfexport'
    export_version = 1
    export_path = os.path.join(tf.compat.as_bytes(export_base), tf.compat.as_bytes(str(export_version)))
    
    builder = saved_model_builder.SavedModelBuilder(export_path)
    print(yolo.model.inputs[0])
    signature = predict_signature_def(
                    inputs={
                            "input_image": yolo.model.inputs[0], 
                            "true_boxes": yolo.model.inputs[1]
                            },
                    outputs={"outputs": yolo.model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess,
            tags=[tag_constants.SERVING], 
            signature_def_map={'predict': signature})
        builder.save()

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
