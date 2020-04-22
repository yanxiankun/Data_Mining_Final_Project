"""
This implementation is based on the fllowing reference code:
https://github.com/joeddav/devol

Xiankun Yan 11/24/2019

"""

from __future__ import print_function

import sys
sys.path.append('.\\')


import random as rand
import csv
import operator
import gc
import os
from datetime import datetime
from keras.callbacks import EarlyStopping
from keras.models import load_model
import keras.backend as K
from sklearn.metrics import log_loss
import numpy as np
import noisyAndLayer

if K.backend() == 'tensorflow':
    import tensorflow as tf

__all__ = ['DEvol']

METRIC_OPS = [operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [min, max]


class DEvol:
    """
    Object which carries out genetic search and returns top performing model
    upon completion.
    """

    def __init__(self, genome_handler, data_path=""):
        """
        Initialize a DEvol object which carries out the training and evaluation
        of a genetic search.

        Args:
            genome_handler (GenomeHandler): the genome handler object defining
                    the restrictions for the architecture search space
            data_path (str): the file which the genome encodings and metric data
                    will be stored in
        """
        self.genome_handler = genome_handler
        self.datafile = data_path or ('test.csv')
        self._bssf = -1

        if os.path.isfile(data_path) and os.stat(data_path).st_size > 1:
            raise ValueError(('Non-empty file %s already exists. Please change'
                              'file path to prevent overwritten genome data.'
                              % data_path))

        print("Genome encoding and metric data stored at", self.datafile, "\n")
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            metric_cols = ["Val Loss", "Val Accuracy"]
            genome = genome_handler.genome_representation() + metric_cols
            writer.writerow(genome)

    def set_objective(self, metric):
        """
        Set the metric for optimization. Can also be done by passing to
        `run`.

        Args:
            metric (str): either 'acc' to maximize classification accuracy, or
                    else 'loss' to minimize the loss function
        """
        if metric == 'acc':
            metric = 'accuracy'
        if metric not in ['loss', 'accuracy']:
            raise ValueError(('Invalid metric name {} provided - should be'
                              '"accuracy" or "loss"').format(metric))
        self._metric = metric
        self._objective = "max" if self._metric == "accuracy" else "min"
        self._metric_index = 1 if self._metric == 'loss' else -1
        self._metric_op = METRIC_OPS[self._objective == 'max']
        self._metric_objective = METRIC_OBJECTIVES[self._objective == 'max']

    def run(self, dataset, num_generations, pop_size, epochs, fitness=None,
            metric='accuracy'):
        """
        Run genetic search on dataset given number of generations and
        population size

        Args:
            dataset : tuple or list of numpy arrays in form ((train_data,
                    train_labels), (validation_data, validation_labels))
            num_generations (int): number of generations to search
            pop_size (int): initial population size
            epochs (int): epochs for each model eval, passed to keras model.fit
            fitness (None, optional): scoring function to be applied to
                    population scores, will be called on a numpy array which is
                    a min/max scaled version of evaluated model metrics, so It
                    should accept a real number including 0. If left as default
                    just the min/max scaled values will be used.
            metric (str, optional): must be "accuracy" or "loss" , defines what
                    to optimize during search

        Returns:
            keras model: best model found with weights
        """
        self.set_objective(metric)

        # If no validation data is given set it to None
        if len(dataset) == 2:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
            self.x_val = None
            self.y_val = None
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_val, self.y_val) = dataset

        # generate and evaluate initial population
        members = self._generate_random_population(pop_size)
        pop = self._evaluate_population(members,
                                        epochs,
                                        fitness,
                                        0,
                                        num_generations)

        # evolve
        for gen in range(1, num_generations):
            members = self._reproduce(pop, gen)
            pop = self._evaluate_population(members,
                                            epochs,
                                            fitness,
                                            gen,
                                            num_generations)

        return load_model('best-model.h5',  custom_objects={'noisyand': noisyAndLayer.noisyand})

    def _reproduce(self, pop, gen):
        """
        Reproduce new population by the crossover and mutation.
        Here, I proposed the Elite Protection to keep stable in evolution.

        :param pop: the population
        :param gen: the generation
        :return: the new population

        Xiankun Yan modified in 11/26/2019
        """
        members = []

        # 95% of population from crossover
        for _ in range(int(len(pop) * 0.5)):
            members.append(self._crossover(pop.select(), pop.select()))
        # randomly mutate
        for imem, mem in enumerate(members):
            members[imem] = self._mutate(mem, gen)

        # best models survive automatically
        members += pop.get_best(len(pop) - int(len(pop) * 0.5))

        return members

    def _evaluate(self, genome, epochs):
        """
        Train network only one time. That the backpropogation dose work only one time in here.

        :param genome: the network genome
        :param epochs: the epochs
        :return:

        Xiankun Yan made improvement in 11/26/2019
        """
        model = self.genome_handler.decode(genome)
        loss, accuracy = None, None
        fit_params = {
            'x': self.x_train,
            'y': self.y_train,
            'batch_size': 32,
            'validation_split': 0.1,
            'epochs': epochs,
            'verbose': 1,
            'callbacks': [
                EarlyStopping(monitor='val_loss', patience=1, verbose=1)
            ]
        }

        if self.x_val is not None:
            fit_params['validation_split'] = 0.0
            fit_params['shuffle'] = 'batch'
            fit_params['validation_data'] = (self.x_val, self.y_val)

        try:
            model.fit(**fit_params)
            loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        except Exception as e:
            loss, accuracy = self._handle_broken_model(model, e)

        self._record_stats(model, genome, loss, accuracy)

        return model, loss, accuracy

    def _record_stats(self, model, genome, loss, accuracy):
        """
        Record the best model to .csv file

        :param model: network weight
        :param genome: the network genome
        :param loss: loss of network
        :param accuracy: accuracy of network
        :return:

        Xiankun Yan 11/26/2019
        """
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = list(genome) + [loss, accuracy]
            writer.writerow(row)

        met = loss if self._metric == 'loss' else accuracy
        if (self._bssf is -1 or
                self._metric_op(met, self._bssf) and
                accuracy is not 0):
            try:
                os.remove('best-model.h5')
            except OSError:
                pass
            self._bssf = met
            model.save('best-model.h5')

    def _handle_broken_model(self, model, error):
        """
        Deal with the uncorrect the model

        :param model: CNNs network
        :param error: The error
        :return: Loss and accuracy

        Xiankun Yan 11/27/2019
        """
        del model

        n = self.genome_handler.n_classes
        # loss = log_loss(np.concatenate(([1], np.zeros(n - 1))), np.ones(n) / n)
        loss = float('inf')
        accuracy = 0.0
        gc.collect()

        if K.backend() == 'tensorflow':
            K.clear_session()
            tf.reset_default_graph()

        print('An error occurred and the model could not train:')
        print(error)
        print(('Model assigned poor score. Please ensure that your model'
               'constraints live within your computational resources.'))
        return loss, accuracy

    def _evaluate_population(self, members, epochs, fitness, igen, ngen):
        """

        Evaluate the population.

        Xiankun Yan 11/27/2019
        """
        fit = []
        for imem, mem in enumerate(members):
            self._print_evaluation(imem, len(members), igen, ngen)
            res = self._evaluate(mem, epochs)
            v = res[self._metric_index]
            del res
            fit.append(v)

        fit = np.array(fit)
        self._print_result(fit, igen)
        return _Population(members, fit, fitness, obj=self._objective)

    def _print_evaluation(self, imod, nmod, igen, ngen):
        fstr = '\nmodel {0}/{1} - generation {2}/{3}:\n'
        print(fstr.format(imod + 1, nmod, igen + 1, ngen))

    def _generate_random_population(self, size):
        return [self.genome_handler.generate() for _ in range(size)]

    def _print_result(self, fitness, generation):
        result_str = ('Generation {3}:\t\tbest {4}: {0:0.4f}\t\taverage:'
                      '{1:0.4f}\t\tstd: {2:0.4f}')
        print(result_str.format(self._metric_objective(fitness),
                                np.mean(fitness),
                                np.std(fitness),
                                generation + 1, self._metric))

    def _crossover(self, genome1, genome2):
        """
        Crossover operation

        Xiankun Yan 11/27/2019
        """
        cross_ind = rand.randint(0, len(genome1))
        child = genome1[:cross_ind] + genome2[cross_ind:]
        return child

    def _mutate(self, genome, generation):
        """

        Mutation operation

        Xiankun Yan  modified in 11/27/2019

        """

        # increase mutations as program continues
        num_mutations = max(3, generation // 4)
        return self.genome_handler.mutate(genome, num_mutations)


class _Population(object):
    """
    Manage population operation: Get the best individual and selection operation.

    Xiankun Yan 11/27/2019

    """

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, score, obj='max'):
        self.members = members
        scores = fitnesses - fitnesses.min()
        if scores.max() > 0:
            scores /= scores.max()
        if obj == 'min':
            scores = 1 - scores
        if score:
            self.scores = score(scores)
        else:
            self.scores = scores
        self.s_fit = sum(self.scores)

    def get_best(self, n):
        """
        selection scheme: Stochastic ranking.

        :param n: the selection number
        :return: some genomes

        Xiankun Yan  imporvement in11/28/2019
        """

        p_t = 0.47
        u = np.random.random()

        for i in range(len(self.scores)):
            for j in range(0, n - i - 1):
                if self.scores[j] < self.scores[j + 1] or u < p_t:
                    self.scores[j], self.scores[j + 1] = self.scores[j + 1], self.scores[j]
                    self.members[j], self.members[j + 1] = self.members[j + 1], self.members[j]

        combined = [(self.members[i], self.scores[i])
                    for i in range(len(self.members))]
        #sorted(combined, key=(lambda x: x[1]), reverse=True)
        return [x[0] for x in combined[:n]]

    def select(self):
        """
        Selection operation

        Xiankun Yan 11/28/2019
        """
        dart = rand.uniform(0, self.s_fit)
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.scores[i]
            if sum_fits >= dart:
                return self.members[i]
