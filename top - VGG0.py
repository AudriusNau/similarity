import tensorflow as tf
import numpy as np

from time import time
from PIL import Image
from tqdm import tqdm

from tensorflow import saved_model
from tensorflow.keras import backend, applications, optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Reshape
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.python.framework import ops

import argparse
import json, h5py, os, shutil, sys
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("TensorFlow version: " + tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--train-model', dest = 'train_model', action = 'store_true')
parser.add_argument('-s', '--test-model', dest = 'test_model', action = 'store_true')
parser.add_argument('-l', '--lc', dest = 'lc', type = int, default = 8)
parser.add_argument('-b', '--batch-size', dest = 'batch_size', type = int, default = 48)
parser.add_argument('-e', '--epochs', dest = 'epochs', type = int, default = 50)

args = parser.parse_args()

checkpoint_dir = 'runtime_files'
checkpoint_auto_dir = 'runtime_files/saved_model.h5'

train_dir_a = 'data/audrius_data/train_a'
valid_dir_a = 'data/audrius_data/valid_a'

train_dir_p = 'data/audrius_data/train_p'
valid_dir_p = 'data/audrius_data/valid_p'

train_dir_n = 'data/audrius_data/train_n'
valid_dir_n = 'data/audrius_data/valid_n'

test_dir_a = 'data/audrius_data/test_a'
test_dir_p = 'data/audrius_data/test_p'
test_dir_n = 'data/audrius_data/test_n'

argN = 64
batch_size=48

def triplet_loss(N = argN, epsilon = 1e-6):
  def triplet_loss(y_true, y_pred):
    beta = N
    print(y_pred.get_shape())
   
    
    anchor = y_pred[0::3]
    positive = y_pred[1::3]
    negative = y_pred[2::3]

    positive_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, positive)), axis = 0, keepdims = True)
    negative_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, negative)), axis = 0, keepdims = True)

    # -ln(-x/N+1)
    positive_distance = -tf.math.log(-tf.math.divide((positive_distance), beta) + 1 + epsilon)
    negative_distance = -tf.math.log(-tf.math.divide((N - negative_distance), beta) + 1 + epsilon)

    loss = negative_distance + positive_distance
    return loss
  return triplet_loss

def pd(N = argN, epsilon = 1e-6):
  def pd(y_true, y_pred):
    beta = N
    anchor = y_pred[0::3]
    positive = y_pred[1::3]
    positive_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, positive)), axis=0)
    positive_distance = -tf.math.log(-tf.math.divide((positive_distance), beta) + 1 + epsilon)
    return backend.mean(positive_distance)
  return pd

def nd(N = argN, epsilon = 1e-6):
  def nd(y_true, y_pred):
    beta = N
    anchor = y_pred[0::3]
    negative = y_pred[2::3]
    negative_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, negative)), axis=0)
    negative_distance = -tf.math.log(-tf.math.divide((N - negative_distance), beta) + 1 + epsilon)
    return backend.mean(negative_distance)
  return nd

def make_model():
  input_a = Input(shape = (224, 224, 3),  name = 'input_a')
  input_p = Input(shape = (224, 224, 3),  name = 'input_p')
  input_n = Input(shape = (224, 224, 3),  name = 'input_n')

  base_model = applications.VGG16(include_top = False, weights = 'imagenet')  #-except the last clasification layer

#  model_a = Sequential()
  
  vgg16_a = base_model(input_a)
  vgg16_p = base_model(input_p)
  vgg16_n = base_model(input_n)

# =============================================================================
#   l1_a = base_model.layers[0](input_a)
#   l1_p = base_model.layers[0](input_p)
#   l1_n = base_model.layers[0](input_n)
#   
#   l2_a = base_model.layers[1](l1_a)
#   l2_p = base_model.layers[1](l1_p)
#   l2_n = base_model.layers[1](l1_n)
#   
#   l3_a = base_model.layers[2](l2_a)
#   l3_p = base_model.layers[2](l2_p)
#   l3_n = base_model.layers[2](l2_n)
# 
#   l4_a = base_model.layers[3](l3_a)
#   l4_p = base_model.layers[3](l3_p)
#   l4_n = base_model.layers[3](l3_n)
# 
#   l5_a = base_model.layers[4](l4_a)
#   l5_p = base_model.layers[4](l4_p)
#   l5_n = base_model.layers[4](l4_n)
# 
#   l6_a = base_model.layers[5](l5_a)
#   l6_p = base_model.layers[5](l5_p)
#   l6_n = base_model.layers[5](l5_n)
# 
#   l7_a = base_model.layers[6](l6_a)
#   l7_p = base_model.layers[6](l6_p)
#   l7_n = base_model.layers[6](l6_n)
# 
#   l8_a = base_model.layers[7](l7_a)
#   l8_p = base_model.layers[7](l7_p)
#   l8_n = base_model.layers[7](l7_n)
# 
#   l9_a = base_model.layers[8](l8_a)
#   l9_p = base_model.layers[8](l8_p)
#   l9_n = base_model.layers[8](l8_n)
# =============================================================================

  #lt1 = tf.keras.layers.concatenate([l9_a, l9_p, l9_n], name='main_concat')

  dense1  = Dense(100, activation='sigmoid')
  drop1   = Dropout(0.4)
  dense2  = Dense(16, activation='sigmoid')
   
#  lt1 = Dense(64, activation = 'sigmoid')
#  lt2 = Dropout(0.5)
#  lt3 = Dense(8, activation = 'sigmoid')

  lt1_a = dense1(vgg16_a)
  lt1_p = dense1(vgg16_p)
  lt1_n = dense1(vgg16_n)

  lt2_a = drop1(lt1_a)
  lt2_p = drop1(lt1_p)
  lt2_n = drop1(lt1_n)

  lt3_a = dense2(lt2_a)
  lt3_p = dense2(lt2_p)
  lt3_n = dense2(lt2_n)

  output = tf.keras.layers.concatenate([lt3_a, lt3_p, lt3_n], axis = 0, name = 'outxxx')

  model = tf.keras.models.Model(inputs = [input_a, input_p, input_n], outputs = output)
 
  for layer in model.layers:
    layer.trainable = False   # freeze all layers until
    if layer.name == 'dense': # the first Dense layer and exit
      layer.trainable = True   
      break
    

  model.compile(optimizer = optimizers.Adam(), loss = triplet_loss(), metrics = [pd(), nd()])

  return model

def train_generator_triplet():
  gen = ImageDataGenerator()
  gen_a = gen.flow_from_directory(directory = train_dir_a, target_size = (224, 224), batch_size = batch_size, class_mode = 'categorical', shuffle = False)
  gen_p = gen.flow_from_directory(directory = train_dir_p, target_size = (224, 224), batch_size = batch_size, class_mode = 'categorical', shuffle = False)
  gen_n = gen.flow_from_directory(directory = train_dir_n, target_size = (224, 224), batch_size = batch_size, class_mode = 'categorical', shuffle = True)
  while True:
    an = gen_a.next()
    po = gen_p.next()
    ne = gen_n.next()
    yield [an[0], po[0], ne[0]], an[1]

def valid_generator_triplet():
  gen = ImageDataGenerator()
  gen_a = gen.flow_from_directory(directory = valid_dir_a, target_size = (224, 224), batch_size = batch_size, class_mode = 'categorical', shuffle = False)
  gen_p = gen.flow_from_directory(directory = valid_dir_p, target_size = (224, 224), batch_size = batch_size, class_mode = 'categorical', shuffle = False)
  gen_n = gen.flow_from_directory(directory = valid_dir_n, target_size = (224, 224), batch_size = batch_size, class_mode = 'categorical', shuffle = True)
  while True:
    an = gen_a.next()
    po = gen_p.next()
    ne = gen_n.next()
    yield [an[0], po[0], ne[0]], an[1]

def test_generator_triplet():
  gen = ImageDataGenerator()
  gen_a = gen.flow_from_directory(directory = test_dir_a, target_size = (224, 224), batch_size = batch_size, class_mode = 'categorical', shuffle = False)
  gen_p = gen.flow_from_directory(directory = test_dir_p, target_size = (224, 224), batch_size = batch_size, class_mode = 'categorical', shuffle = False)
  gen_n = gen.flow_from_directory(directory = test_dir_n, target_size = (224, 224), batch_size = batch_size, class_mode = 'categorical', shuffle = True)
  while True:
    an = gen_a.next()
    po = gen_p.next()
    ne = gen_n.next()
    yield [an[0], po[0], ne[0]], an[1]

def train_model():
  model = make_model()
  print(model.summary())
  #datagen = ImageDataGenerator()
  #train_generator = datagen.flow_from_directory(directory = args.train_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = 'categorical', shuffle = False)
  #valid_generator = datagen.flow_from_directory(directory = args.valid_dir, target_size = (224, 224), batch_size = args.batch_size, class_mode = 'categorical', shuffle = False)
  cb_tensorboard = TensorBoard(log_dir = "logs", histogram_freq = 2, write_graph = True, write_images = True)

 # cb_tensorboard = TensorBoard(log_dir = "/runtime_files/logs/{}".format(time()), histogram_freq = 2, write_graph = True, write_images = True)
  cb_checkpoint = ModelCheckpoint(checkpoint_auto_dir, save_weights_only = False, period = 100, verbose = 1)

  model.fit_generator(generator = train_generator_triplet(), steps_per_epoch = argN, epochs = args.epochs, validation_data = valid_generator_triplet(), validation_steps = 3, callbacks = [cb_tensorboard, cb_checkpoint])
  
  #tf.keras.experimental.export_saved_model(model, checkpoint_dir)
#  tf.saved_model.save(model, checkpoint_dir)
  tf.keras.models.save_model(model, checkpoint_dir) # checkpoint_dir
 
def test_model():
  test_samples = len(os.listdir(test_dir_a + "/0"))
  
  #datagen = ImageDataGenerator()
  #test_generator = datagen.flow_from_directory(directory = args.test_dir, target_size = (224, 224), batch_size = test_samples, class_mode = 'categorical', shuffle = False)
  
#  model = tf.keras.experimental.load_from_saved_model(checkpoint_dir)
  
#  model = tf.keras.models.load_model(checkpoint_dir)
  
  model = tf.saved_model.load('runtime_files')
  #model = tf.saved_model.load('models_history/runtime_files_100_16') 
  infer = model.signatures["serving_default"]

  i=0
  pbar = tqdm(total=test_samples)
  for x, _ in test_generator_triplet():
    i=i+1
    if i%10==0 :
      pbar.update(10)
    #print(i)
    prediction=infer(input_a=tf.constant(x[0]), input_p=tf.constant(x[1]), input_n=tf.constant(x[2]))
    outxxx=prediction['outxxx']
    try:
      results =np.concatenate((results, outxxx), axis=0)
    except NameError:
      results = outxxx

    if i>=test_samples : break #test_samples
  pbar.close()
  #model = make_model()
  #model.load_weights(checkpoint_auto_dir)

  #results = model.predict_generator(generator = test_generator_triplet(), steps = test_samples, verbose = 0)
  #results = model.predict(generator = test_generator_triplet(), steps = test_samples, verbose = 0)
  
  N = argN
  beta = N
  epsilon = 1e-6

  anchor = results[0::3]
  positive = results[1::3]
  negative = results[2::3]

  positive_distance = np.nansum(np.square(anchor - positive), axis = 1)
  positive_distance = - np.log(- (positive_distance / beta) + 1 + epsilon)

  positive_distance2 = np.nansum(np.square(anchor - negative), axis = 1)
  positive_distance2 = - np.log(- (positive_distance2 / beta) + 1 + epsilon)
 
  tp = 0
  fp = 0
  pneq = 0
  min_p = sys.maxsize
  max_p = 0
  min_n = sys.maxsize
  max_n = 0

  for i in range(test_samples): #test_samples
    pda = np.nansum(positive_distance[i])
    nda = np.nansum(positive_distance2[i])
    print(pda, "\t", nda)
    if pda >= nda:
    # print(i)
      fp += 1
    else:
      tp += 1
    if pda == nda:
      pneq += 1

    if min_p > pda:
      min_p = pda
    if max_p < pda:
      max_p = pda

    if min_n > nda:
      min_n = nda
    if max_n < nda:
      max_n = nda

  print(min_p, ' - ', max_p, ', ', min_n, ' - ', max_n)
  print('accuracy: ', np.round(tp / (tp + fp) * 100, 1))
  print('equal predictions: ', pneq)
  
  import matplotlib.pyplot as plt
  plt.style.use('seaborn-whitegrid')
  for x in range(0, test_samples):
      y_p = np.nansum(positive_distance[x])
      plt.plot(x, y_p, 'x', color='blue');
      y_n = np.nansum(positive_distance2[x])
      plt.plot(x, y_n, 'x', color='red');
#shutil.rmtree(checkpoint_dir, ignore_errors = True)  # remove runtime directory

#model = make_model()
#print(model.summary())

#train_model()
test_model()




