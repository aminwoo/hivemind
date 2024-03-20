import numpy as np 
import tensorflow as tf
import tqdm

import time 
start = time.time()


data = np.load("data/fics_training_data/checkpoint0.npz")
dataset = tf.data.Dataset.from_tensor_slices((data["board_planes"], data["move_planes"], data["value_planes"]))
dataset = dataset.shuffle(buffer_size=2**16).batch(1024)
print(dataset.cardinality().numpy())
for x, _, _ in tqdm.tqdm(dataset):
    print(x.shape)

print(time.time() - start)