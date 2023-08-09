import tensorflow as tf

from transformer import Transformer
from tiny_shakespeare import TinyShakespeare

checkpoint_path = "checkpoints/cp-0010.ckpt"


custom_objects = {'Transformer': Transformer}
model = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects)

dataset = TinyShakespeare(decoder='simple')

start = tf.zeros((1, 1), dtype=tf.int64)
print(dataset.decode(model.generate(start, 2000)[0].numpy().tolist()))