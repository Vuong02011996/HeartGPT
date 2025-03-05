import os

path_origin = "/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/"
path_model = path_origin + 'Model/'
path_model_ec57 = path_origin + 'Model_EC57/'
if not os.path.exists(path_model):
    os.makedirs(path_model)
if not os.path.exists(path_model_ec57):
    os.makedirs(path_model_ec57)

path_save = path_origin + 'Data/Data_Study_N_V/'


#define vocab size. All data was scaled between 0 and 100 and rounded to nearest integer, giving 101 possible token values
# Q/A if don't convert signal  to 0-> 100, what the vocab_size
# vocab_size = 101
vocab_size = 1001
block_size = 30 # this is context length
before, after = 12, 18

num_classes = 2


save_log_path = '/home/server2/Desktop/Vuong/Data/PhysionetData/'
path2db = save_log_path + 'mitdb'

