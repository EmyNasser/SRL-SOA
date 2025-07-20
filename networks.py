import tensorflow as tf
import numpy as np

from layers import Oper1D

np.random.seed(10)
tf.random.set_seed(10)
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf

def SLRol(n_bands, q):
    input = Input(shape=(n_bands,))
    x_0 = tf.Variable(tf.random.normal((n_bands, q)), trainable=True)

    # Wrap tf.matmul in a Lambda layer
    y = Lambda(lambda x: tf.matmul(x, x_0))(input)

    model = Model(inputs=input, outputs=y)
    return model
