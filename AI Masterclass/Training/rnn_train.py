# Training the MDN-RNN model

# Importing the libraries

import numpy as np
import os
import json
import time
from vae import reset_graph
from rnn import HyperParams, MDNRNN

# Setting the index of what GPUs (if available) to use in the training process
os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

# Defining the DATA_DIR variable which points to the folder where the RNN training data is stored
DATA_DIR = "series"
# Checks if save folder for RNN weights exisits, if not it will crate one
model_save_path = "tf_rnn"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

# Checking if the folder storing the latent vectors exists, and if not, creating one
initial_z_save_path = "tf_initial_z"
if not os.path.exists(initial_z_save_path):
  os.makedirs(initial_z_save_path)

# Making a function that returns a random batch of latent vectors and actions
def random_batch():
  indices = np.random.permutation(N_data)[0:batch_size]
  mu = data_mu[indices]
  logvar = data_logvar[indices]
  action = data_action[indices]
  s = logvar.shape
  z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
  return z, action

# Making a function that returns all the default hyperparameters of the MDN-RNN model

def default_hps():
  return HyperParams(num_steps=4000,
                     max_seq_len=999,
                     input_seq_width=35,
                     output_seq_width=32,
                     rnn_size=256,
                     batch_size=100,
                     grad_clip=1.0,
                     num_mixture=5,
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0,
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)

# Getting and sampling these default hyperparameters

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)
raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))

# Splitting the training data into three separate specific parts (mu, logvar and action)

data_mu = raw_data["mu"]
data_logvar = raw_data["logvar"]
data_action =  raw_data["action"]

# Setting the hyperparameters used for data batching and model training

max_seq_len = hps_model.max_seq_len
N_data = len(data_mu)
batch_size = hps_model.batch_size

# Saving 1000 initial mu and logvar from the above data split

initial_mu = np.copy(data_mu[:1000, 0, :]*10000).astype(np.int).tolist()
initial_logvar = np.copy(data_logvar[:1000, 0, :]*10000).astype(np.int).tolist()
with open('initial_z.json', 'wt') as outfile:
  json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))

# Resetting the graph of the MDN-RNN model

reset_graph()

# Creating the MDN-RNN model as an object of the MDNRNN class with all the default hyperparameters

rnn = MDNRNN(hps_model)

# Implementing the Training Loop

hps = hps_model
start = time.time()
for local_step in range(hps.num_steps):
  step = rnn.sess.run(rnn.global_step)
  curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate
  raw_z, raw_a = random_batch()
  inputs = np.concatenate((raw_z[:, :-1, :], raw_a[:, :-1, :]), axis=2)
  outputs = raw_z[:, 1:, :]
  feed = {rnn.input_x: inputs, rnn.output_x: outputs, rnn.lr: curr_learning_rate}
  (train_cost, state, train_step, _) = rnn.sess.run([rnn.cost, rnn.final_state, rnn.global_step, rnn.train_op], feed)
  if (step%20==0 and step > 0):
    end = time.time()
    time_taken = end-start
    start = time.time()
    output_log = "Step: %d, Learning Rate: %.6f, Cost: %.4f, Training Time: %.4f" % (step, curr_learning_rate, train_cost, time_taken)
    print(output_log)

# Saving the weights of the MDN-RNN model into a json file

rnn.save_json(os.path.join(model_save_path, "rnn.json"))
