"""impsy.convert: Functions for convert an impsy mdrnn model to the .tflite format"""

import click
import keras_mdn_layer as mdn
from .utils import mdrnn_config
import re

import impsy.mdrnn as mdrnn
from tensorflow import keras
import tensorflow as tf




@click.command(name="convert2")
@click.option("-D", "--directory", type=str, default="models/", help="The directory to find the keras file.")
@click.option("-F", "--filepath", type = str, default="musicMDRNN-dim9-layers2-units16-mixtures5-scale10-sm.keras", help="The name of the .keras file to be converted")
def convert2(directory: str, filepath: str):
    """Loads a .keras mdrnn model and converts it to a .tflite model."""
    
    # Old methods to save 
    # model.save_weights(model_dir + model_name + ".h5")
    # tflite_model_name = f'{model_dir}{model_name}-sm.keras'
    # model.save(tflite_model_name, save_format='keras')
    
    
    # Getting mixtures from filename
    def extract_mixtures(file_name):
        match = re.search(r'mixtures(\d+)', file_name)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("Cannot find the mixture amount in the file name.")
    
    N_MIXES = extract_mixtures(filepath)
    
    # Load the model
    tflite_model_name = directory + filepath
    model = tf.keras.models.load_model(tflite_model_name, custom_objects={'MDN': mdn.MDN, 'mdn_loss_func': mdn.get_mixture_loss_func(1, N_MIXES)})
    print("Successfully loaded... \n")
    
    
    # Converts the filepath name into blank name so we can add tflite
    def remove_keras_extension(filepath):
        if filepath.endswith(".keras"):
            return filepath[:-6]
        else:
            raise ValueError("Invalid file extension. Expected '.keras' at the end of the filepath.")
    
    """"
    ## Converting for tensorflow lite.
    # Converting in progress
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS, # enable TensorFlow ops.   
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    tflite_model_name = directory + remove_keras_extension(filepath) + '-lite.tflite'
    with open(tflite_model_name, 'wb') as f:
        f.write(tflite_model)

    print("Conversion done, bye.")
    """
    # Model Hyperparameters
    SEQ_LEN = 50

    # Directly defining "training" function's inputs
    dimension = 9
    batchsize = 64
    
    BATCH_SIZE = batchsize
    STEPS = SEQ_LEN
    INPUT_SIZE = dimension 
    
    
    
    # Attempting colab conversion 
    run_model = tf.function(lambda x: model(x))
    # This is important, let's fix the input size.
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))

    # model directory.
    MODEL_DIR = "models"
    model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    tflite_model = converter.convert()
    
    print("Conversion done, bye.")