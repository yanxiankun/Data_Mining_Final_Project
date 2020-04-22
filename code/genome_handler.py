"""
This implementation is based on the fllowing reference code:
https://github.com/joeddav/devol

Xiankun Yan 11/20/2019

"""

import sys
sys.path.append('.\\')

import numpy as np
import random as rand
import math
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import noisyAndLayer

class GenomeHandler:
    """
    Defines the configuration and handles the conversion and mutation of
    individual genomes. Should be created and passed to a `DEvol` instance.

    ---
    Genomes are represented as fixed-with lists of integers corresponding
    to sequential layers and properties. A model with 2 convolutional layers
    and 1 dense layer would look like:

    [<conv layer><conv layer><dense layer><optimizer>]

    The makeup of the convolutional layers and dense layers is defined in the
    GenomeHandler below under self.convolutional_layer_shape and
    self.dense_layer_shape. <optimizer> consists of just one property.
    """

    def __init__(self, max_conv_layers, max_dense_layers, max_filters,
                 max_dense_nodes, input_shape, n_classes,
                 batch_normalization=True, dropout=True, max_pooling=True,
                 optimizers=None, activations=None):
        """
        Creates a GenomeHandler according 

        Args:
            max_conv_layers: The maximum number of convolutional layers           
            max_conv_layers: The maximum number of dense (fully connected)
                    layers, including output layer
            max_filters: The maximum number of conv filters (feature maps) in a
                    convolutional layer
            max_dense_nodes: The maximum number of nodes in a dense layer
            input_shape: The shape of the input
            n_classes: The number of classes
            batch_normalization (bool): whether the GP should include batch norm
            dropout (bool): whether the GP should include dropout
            max_pooling (bool): whether the GP should include max pooling layers
            optimizers (list): list of optimizers to be tried by the GP. By
                    default, the network uses Keras's built-in adam, rmsprop,
                    adagrad, and adadelta
            activations (list): list of activation functions to be tried by the
                    GP. By default, relu and sigmoid.
        """
        if max_dense_layers < 1:
            raise ValueError(
                "At least one dense layer is required for softmax layer"
            ) 
        if max_filters > 0:
            filter_range_max = int(math.log(max_filters, 2)) + 1
        else:
            filter_range_max = 0
        self.optimizer = optimizers or [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta'
        ]
        self.activation = activations or [
            'relu',
            'sigmoid',
        ]
        self.convolutional_layer_shape = [
            "active",
            "num filters",
            "batch normalization",
            "activation",
            "dropout",
            "max pooling",
        ]
        self.dense_layer_shape = [
            "active",
            "num nodes",
            "batch normalization",
            "activation",
            "dropout",
        ]
        self.layer_params = {
            "active": [0, 1],
            "num filters": [2**i for i in range(3, filter_range_max)],
            "num nodes": [2**i for i in range(4, int(math.log(max_dense_nodes, 2)) + 1)],
            "batch normalization": [0, (1 if batch_normalization else 0)],
            "activation": list(range(len(self.activation))),
            "dropout": [(i if dropout else 0) for i in range(11)],
            "max pooling": list(range(3)) if max_pooling else 0,
        }

        self.convolution_layers = max_conv_layers
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        self.dense_layers = max_dense_layers - 1 # this doesn't include the softmax layer, so -1
        self.dense_layer_size = len(self.dense_layer_shape)
        self.input_shape = input_shape
        self.n_classes = n_classes

    def convParam(self, i):
        """
        Get the shape of convolution layer

        :param i: the nth layer
        :return: the different parameter in  convolutional_layer_shape

        Xiankun Yan 11/20/2019
        """
        key = self.convolutional_layer_shape[i]
        return self.layer_params[key]

    def denseParam(self, i):
        """

        Get the shape of dense layer

        :param i: the nth layer
        :return: the different parameter in  dense_layer_shape

        Xiankun Yan 11/20/2019
        """

        key = self.dense_layer_shape[i]
        return self.layer_params[key]

    def mutate(self, genome, num_mutations):
        """
        mutation operation: We mutate the gene segments on the genome


        :param genome: the CNNs Genome
        :param num_mutations: the number of gene segments that need to mutate
        :return: mutated CNNs Genome

        Xiankun Yan  modified in 11/20/2019
        """
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            index = np.random.choice(list(range(1, len(genome))))
            if index < self.convolution_layer_size * self.convolution_layers:
                if genome[index - index % self.convolution_layer_size]:
                    range_index = index % self.convolution_layer_size
                    choice_range = self.convParam(range_index)
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.3: # randomly flip deactivated layers
                    genome[index - index % self.convolution_layer_size] = 1
            elif index != len(genome) - 1:
                offset = self.convolution_layer_size * self.convolution_layers
                new_index = (index - offset)
                present_index = new_index - new_index % self.dense_layer_size
                if genome[present_index + offset]:
                    range_index = new_index % self.dense_layer_size
                    choice_range = self.denseParam(range_index)
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.3:
                    genome[present_index + offset] = 1
            else:
                genome[index] = np.random.choice(list(range(len(self.optimizer)))) 
        return genome

    def decode(self, genome):
        """
        Decoding: The genome is a language that describes the CNNs model. According to language, we build
        the CNNs model.


        :param genome: The CNNs genome
        :return:The CNNs model

        Xiankun Yan 11/22/2019
        """
        if not self.is_compatible_genome(genome):
            raise ValueError("Invalid genome for specified configs")
        model = Sequential()
        dim = 0
        offset = 0
        if self.convolution_layers > 0:
            dim = min(self.input_shape[:-1]) # keep track of smallest dimension
        input_layer = True
        for i in range(self.convolution_layers):
            if genome[offset]:
                convolution = None
                if input_layer:
                    convolution = Convolution2D(
                        genome[offset + 1], (3, 3),
                        padding='same',
                        input_shape=self.input_shape
                    )
                    input_layer = False
                else:
                    convolution = Convolution2D(
                        genome[offset + 1], (3, 3),
                        padding='same'
                    )
                model.add(convolution)
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                model.add(Activation(self.activation[genome[offset + 3]]))
                model.add(Dropout(float(genome[offset + 4] / 20.0)))
                max_pooling_type = genome[offset + 5]
                # must be large enough for a convolution
                if max_pooling_type == 1 and dim >= 5:
                    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
                    dim = int(math.ceil(dim / 2))
            offset += self.convolution_layer_size

        if not input_layer:
            #model.add(GlobalAveragePooling2D())
            model.add(noisyAndLayer.noisyand(self.n_classes+1))

        for i in range(self.dense_layers):
            if genome[offset]:
                dense = None
                if input_layer:
                    dense = Dense(genome[offset + 1], input_shape=self.input_shape)
                    input_layer = False
                else:
                    dense = Dense(genome[offset + 1])
                model.add(dense)
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                model.add(Activation(self.activation[genome[offset + 3]]))
                model.add(Dropout(float(genome[offset + 4] / 20.0)))
            offset += self.dense_layer_size
        
        model.add(Dense(self.n_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
            optimizer=self.optimizer[genome[offset]],
            metrics=["accuracy"])
        return model

    def genome_representation(self):
        """
        Encoding operation: Once we get a new CNNs model, we need to encoding it to genome which is the string data.

        :return: The string genome

        Xiankun Yan 11/22/2019
        """
        encoding = []
        for i in range(self.convolution_layers):
            for key in self.convolutional_layer_shape:
                encoding.append("Conv" + str(i) + " " + key)
        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                encoding.append("Dense" + str(i) + " " + key)
        encoding.append("Optimizer")
        return encoding

    def generate(self):
        """
        Generation: At the begin, we need to create the initial genome with random parameters.

        :return: The one individual in population.

        Xiankun Yan 11/22/2019
        """
        genome = []
        for i in range(self.convolution_layers):
            for key in self.convolutional_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))
        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))
        genome.append(np.random.choice(list(range(len(self.optimizer)))))
        genome[0] = 1
        return genome

    def is_compatible_genome(self, genome):
        """
        This function is to determine whether the generated neural network meets the requirements.

        :param genome: One CNNs genome
        :return: True or False

        Xiankun Yan 11/22/2019
        """
        expected_len = self.convolution_layers * self.convolution_layer_size \
                        + self.dense_layers * self.dense_layer_size + 1
        if len(genome) != expected_len:
            return False
        ind = 0
        for i in range(self.convolution_layers):
            for j in range(self.convolution_layer_size):
                if genome[ind + j] not in self.convParam(j):
                    return False
            ind += self.convolution_layer_size
        for i in range(self.dense_layers):
            for j in range(self.dense_layer_size):
                if genome[ind + j] not in self.denseParam(j):
                    return False
            ind += self.dense_layer_size
        if genome[ind] not in range(len(self.optimizer)):
            return False
        return True

    def best_genome(self, csv_path, metric="accuracy", include_metrics=True):
        """
        Get the best CNNs genome.

        :param csv_path: Genome description
        :param metric: The metric on CNNs model
        :param include_metrics: Add metrics behind the CNNs genome.
        :return: The best string genome
        """
        best = max if metric is "accuracy" else min
        col = -1 if metric is "accuracy" else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        return genome

    def decode_best(self, csv_path, metric="accuracy"):
        """

        :param csv_path: Genome description
        :param metric: The metric on CNNs model
        :return: The Best CNNs model
        """
        return self.decode(self.best_genome(csv_path, metric, False))
