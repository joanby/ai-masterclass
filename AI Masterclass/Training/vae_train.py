# Training the VAE model

# Importing the libraries

import os
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
from vae import ConvVAE, reset_graph

# Setting the OS Hyperparameters

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Setting the VAE Hyperparameters

z_size = 32
batch_size = 100
learning_rate = 0.0001
kl_tolerance = 0.5

# Setting the Training Hyperparameters

NUM_EPOCH = 10
DATA_DIR = "record"

# Making a directory to save the weights of the VAE model

model_save_path = "tf_vae"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

# Making a function that returns the number of generated data files (each file represents one game)

def count_length_of_filelist(filelist):
  N = len(filelist)
  total_length = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    total_length += l
    if (i % 1000 == 0):
      print("loading file", i)
  return  total_length

# Making a function that creates a dataset from the saved generated files and returns data which will be the input of the VAE

def create_dataset(filelist, N=10000, M=1000):
  data = np.zeros((M*N, 64, 64, 3), dtype=np.uint8)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    if (idx+l) > (M*N):
      data = data[0:idx]
      print('premature break')
      break
    data[idx:idx+l] = raw_data
    idx += l
    if ((i+1) % 100 == 0):
      print("loading file", i+1)
  return data

# Loading and sorting the first 10000 observations from the created dataset in the "record" folder

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]
dataset = create_dataset(filelist)

# Splitting the dataset into batches

total_length = len(dataset)
num_batches = int(np.floor(total_length/batch_size))
print("num_batches", num_batches)

# Resetting the graph of the VAE model

reset_graph()

# Creating the VAE model as an object of the ConvVAE class

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True)

# Implementing the Training Loop

print("train", "step", "loss", "recon_loss", "kl_loss")
for epoch in range(NUM_EPOCH):
  np.random.shuffle(dataset)
  for idx in range(num_batches):
    batch = dataset[idx*batch_size:(idx+1)*batch_size]
    obs = batch.astype(np.float)/255.0
    feed = {vae.x: obs,}
    (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
      vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
    ], feed)
    if ((train_step+1) % 500 == 0):
      print("step", (train_step+1), train_loss, r_loss, kl_loss)
    if ((train_step+1) % 5000 == 0):
      vae.save_json("tf_vae/vae.json")

# Saving the weights of the VAE model into a json file

vae.save_json("tf_vae/vae.json")
