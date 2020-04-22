import keras
import tensorflow as tf

class noisyand(keras.layers.Layer):
    def __init__(self, num_classes = 2, a = 20, **kwargs):
        self.num_classes = num_classes
        self.a = max(1,a)
        super(noisyand,self).__init__(**kwargs)

    def build(self, input_shape):
        self.b = self.add_weight(name = "b",shape = (1,input_shape[-1]), initializer = "uniform",trainable = True)
        super(noisyand,self).build(input_shape)

    def call(self,x):
        mean = tf.reduce_mean(x, axis = [1,2])
        return (tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)) / (tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]