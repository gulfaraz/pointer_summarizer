import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))#os.path.("~")

#train_data_path = os.path.join(root_dir, "data/finished_files/train.bin")
train_data_path = os.path.join(root_dir, "data", "finished_files", "chunked", "train_*")
eval_data_path = os.path.join(root_dir, "data", "finished_files", "val.bin")
decode_data_path = os.path.join(root_dir, "data", "finished_files", "test.bin")
vocab_path = os.path.join(root_dir, "data", "finished_files", "vocab")
log_root = os.path.join(root_dir, "data", "log")

# Hyperparameters
hidden_dim= 256
glove_dim = 300
elmo_dim = 1024
emb_dim = 1324 # glove_dim + elmo_dim
batch_size= 8
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 10000 #steps (not epochs)

use_gpu=True

lr_coverage=0.15

# For ELMo
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
