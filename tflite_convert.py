import tensorflow as tf 
import numpy as np
import os

model_path = 'models'

converter = tf.lite.TFLiteConverter.from_saved_model(model_path) # path to the SavedModel directory
# converter._enable_tflite_resource_variables = True
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model) 