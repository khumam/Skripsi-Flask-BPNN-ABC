import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as pt
import matplotlib.animation as animation
import math
import random
from tqdm import tqdm
import json


class Bpnn:

    def __init__(self, numInput, numHidden, numOutput, lr=None, activation=None, maxConfig=0, minConfig=0):
        self.numInput = numInput
        self.numHidden = numHidden
        self.numOutput = numOutput

        self.weights_ih = np.random.rand(self.numInput, self.numHidden)
        self.weights_ho = np.random.rand(self.numHidden, self.numOutput)

        self.error_data = []
        self.target_data = []
        self.data_output = []

        self.bias_h = np.random.rand(1, self.numHidden)
        self.bias_o = np.random.rand(1, self.numOutput)

        if lr is not None:
            self.learning_rate = lr
        else:
            self.learning_rate = 0.1

        if activation is not None:
            self.activation = activation
        else:
            self.activation = 'sigmoid'

        self.max_config = maxConfig
        self.min_config = minConfig

    def process(self, inputs):
        inputs = np.array(inputs)
        hidden = np.dot(inputs, self.weights_ih)
        hidden = hidden + self.bias_h

        # Aktivasi
        hidden = self.mapsigmoid(hidden)

        output = np.dot(hidden, self.weights_ho)
        output = output + self.bias_o
        output = self.mapsigmoid(output)

        return output

    def predict(self, inputs, credentials):
        weights_ih = np.array(credentials['weights_ih'])
        weights_ho = np.array(credentials['weights_ho'])
        bias_h = np.array(credentials['bias_h'])
        bias_o = np.array(credentials['bias_o'])

        inputs = np.array(inputs)
        hidden = np.dot(inputs, weights_ih)
        hidden = hidden + bias_h

        # Aktivasi
        hidden = self.mapsigmoid(hidden)

        output = np.dot(hidden, weights_ho)
        output = output + bias_o
        output = self.mapsigmoid(output)

        return output

    def set_weight(self, ih=None, ho=None):
        if ih is not None:
            self.weights_ih = ih
        if ho is not None:
            self.weights_ho = ho

    def get_accuracy(self, root=False):
        mse = self.do_mse(root)
        return 1 - mse

    def get_loss(self, root=False):
        mse = self.do_mse(root)
        return mse

    def sigmoid(self, x):
        return 1 / (1 + pow(math.e, -x))

    def dsigmoid(self, x):
        return x * (1 - x)

    def mapsigmoid(self, data):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                data[i, j] = self.sigmoid(value)
        return data

    def mapdsigmoid(self, data):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                data[i, j] = self.dsigmoid(value)
        return data

    def traindata(self, inputs, targets):
        inputs = np.array(inputs)
        hidden = np.dot(inputs, self.weights_ih)
        hidden = hidden + self.bias_h

        # Aktivasi
        hidden = self.mapsigmoid(hidden)

        output = np.dot(hidden, self.weights_ho)
        output = output + self.bias_o
        outputs = self.mapsigmoid(output)

        self.data_output.append(outputs)

        targets = np.array(targets)
        output_errors = np.subtract(targets, outputs)

        self.generate_error_data(output_errors, targets)

        gradients = self.mapdsigmoid(outputs)
        gradients = np.multiply(output_errors, gradients)
        gradients = gradients * self.learning_rate

        hidden_t = hidden.T
        weight_ho_d = np.dot(hidden_t, gradients)

        self.weights_ho = np.add(self.weights_ho, weight_ho_d)
        self.bias_o = np.add(self.bias_o, gradients)

        who_t = self.weights_ho.T
        hidden_errors = np.dot(output_errors, who_t)

        hidden_gradients = self.mapdsigmoid(hidden)
        hidden_gradients = np.multiply(hidden_errors, hidden_gradients)
        hidden_gradients = hidden_gradients * self.learning_rate

        inputs_t = inputs.T
        weight_ih_d = np.dot(inputs_t, hidden_gradients)

        self.weights_ih = np.add(self.weights_ih, weight_ih_d)
        self.bias_h = np.add(self.bias_h, hidden_gradients)

    def generate_error_data(self, error, target):
        self.error_data.append(error)
        self.target_data.append(target)

    def do_mse(self, root=False):
        sum_error = 0
        total = len(self.error_data)

        for i in range(total):
            self.error_data[i][0][0]**2
            sum_error += self.error_data[i][0][0]

        res = sum_error / total

        if root:
            return math.sqrt(abs(res))
        else:
            return res

    def do_mape(self):
        sum_mape = 0
        total = len(self.target_data)
        for i in range(total):
            sum_mape += (abs(self.target_data[i][0] -
                             self.data_output[i][0][0])) / self.target_data[i][0]
        return sum_mape / total

    def get_weight_ih(self):
        return self.weights_ih

    def get_weight_ho(self):
        return self.weights_ho

    def get_bias_h(self):
        return self.bias_h

    def get_bias_o(self):
        return self.bias_o

    def get_errors(self):
        return np.array(self.error_data)

    def get_targets(self):
        return np.array(self.target_data)

    def get_outputs_data(self):
        return np.array(self.data_output)

    def get_max_config(self):
        return self.max_config

    def get_min_config(self):
        return self.min_config

    def save_model(self, filename):
        model = {}
        model['algorithm'] = []
        model['algorithm'].append(
            'Backpropagation Neural Network')
        model['weights_ih'] = []
        model['weights_ih'].append(self.weights_ih.tolist())
        model['weights_ho'] = []
        model['weights_ho'].append(self.weights_ho.tolist())
        model['bias_h'] = []
        model['bias_h'].append(self.bias_h.tolist())
        model['bias_o'] = []
        model['bias_o'].append(self.bias_o.tolist())
        model['accuracy'] = []
        model['accuracy'].append(self.get_accuracy(True))
        model['loss'] = []
        model['loss'].append(self.get_loss(True))
        model['max_config'] = []
        model['max_config'].append(self.get_max_config())
        model['min_config'] = []
        model['min_config'].append(self.get_min_config())

        with open(filename, 'w') as fileoutput:
            json.dump(model, fileoutput)
