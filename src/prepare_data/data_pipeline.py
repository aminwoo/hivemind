import glob
import time
from random import shuffle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

ARRAY_SHAPES_WITHOUT_BATCH = [(8, 16, 32), (2,), ()]

class Loader:
    """Class to load games in batches."""

    def __init__(self, path: str) -> None:
        self.load(path)

    def load(self, path: str) -> None:
        """
        Load in new batch of games.

        :param data:
        :return:
        """
        #start = time.time()
        self.data = np.load(path) 
        #print(time.time() - start)
        self.idx = 0 

    def get(self, batch_size: int = 1024):
        """
        Get a batch for neural network training.

        :param batch_size: 
        :return:
        """
        start = self.idx * batch_size
        end = (self.idx + 1) * batch_size
        if end > self.data["board_planes"].shape[0]:
            return None 
            
        planes = self.data["board_planes"][start:end]
        policy = self.data["move_planes"][start:end]
        value = self.data["value_planes"][start:end]
        self.idx += 1 

        return planes, policy, value

def data_generator(
        chunk_dir,
        batch_size,
        shuffle_buffer_size,
        validation=False,
):
    assert shuffle_buffer_size % batch_size == 0  
    files = list(glob.glob(chunk_dir + "/*"))
    files = [file for file in files if file.endswith("npz")]
    if len(files) == 0:
        raise FileNotFoundError("No valid input files!")

    if validation:
        files.sort()
    else:
        shuffle(files)

    shuffle_buffers = [
        np.zeros([shuffle_buffer_size] + list(shape)) for shape in ARRAY_SHAPES_WITHOUT_BATCH
    ]
    
    loader = Loader(files.pop())
    while True:
        for i in range(shuffle_buffer_size // batch_size):
            start = time.time()
            processed_batch = loader.get(batch_size)
            print(time.time() - start)
        #    if not processed_batch:
         #       if not files:
         #           break
         #       loader.load(files.pop())
        #        continue
#
            for j in range(len(shuffle_buffers)):
                shuffle_buffers[j][batch_size * i : batch_size * (i  + 1)] = processed_batch[j]

        if not validation:
            for i in range(len(shuffle_buffers)):
                np.random.shuffle(shuffle_buffers[i])

        for i in range(shuffle_buffer_size // batch_size):
            batch = tuple(
                [shuffle_buffer[batch_size * i : batch_size * (i  + 1)] for shuffle_buffer in shuffle_buffers]
            )
            yield batch

def make_callable(chunk_dir, batch_size, shuffle_buffer_size):
    def return_gen():
        return data_generator(
            chunk_dir=chunk_dir,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
        )
    return return_gen

def main():
    batch_size = 1024
    shuffle_buffer_size = 2**12

    gen_callable = make_callable(
        chunk_dir="data/fics_training_data",
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
    )

    array_shapes = [
        tuple([batch_size] + list(shape)) for shape in ARRAY_SHAPES_WITHOUT_BATCH
    ]
    output_signature = tuple(
        [tf.TensorSpec(shape=shape, dtype=tf.float32) for shape in array_shapes]
    )
    gen = tf.data.Dataset.from_generator(
        gen_callable, output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    for board_planes, move_planes, value_planes in tqdm(gen, smoothing=0.01):
        print(board_planes.shape)


if __name__ == "__main__":
    main()
    