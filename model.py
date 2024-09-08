import tensorflow as tf

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Layer

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = tf.python.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.python.keras.layers.MultiheadAttention()