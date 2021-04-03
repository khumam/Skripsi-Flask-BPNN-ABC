from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from Bpnn import Bpnn
from BpnnAbc import BpnnAbc
import random
from datetime import datetime
import matplotlib.pyplot as pt
import json
import math

app = Flask(__name__)


def processDataset(filepath):
    dataset = pd.read_csv(filepath, header=None, nrows=1400, skiprows=1)
    dataset_testing = pd.read_csv(
        filepath, header=None, nrows=600, skiprows=1400)
    data_training = []
    data_testing = []

    for index, rows in dataset.iterrows():
        pre = [rows[10], rows[11], rows[12], rows[13],
               rows[14], rows[15], rows[16], rows[17]]
        tar = [rows[6]]
        data_training.append([pre, tar])

    for index, rows in dataset_testing.iterrows():
        pre = [rows[10], rows[11], rows[12], rows[13],
               rows[14], rows[15], rows[16], rows[17]]
        tar = [rows[6]]
        data_testing.append([pre, tar])

    return data_training, data_testing


def processTrain(dataset_training, dataset_testing, learning_rate, epochs, hidden_neuron, output, type):
    if type == 'bpnn':
        Backpropagation = Bpnn(8, int(hidden_neuron), 1, float(learning_rate))
    else:
        Backpropagation = BpnnAbc(
            8, int(hidden_neuron), 1, float(learning_rate))

    temp_data_training = []
    temp_data_result = []

    for epoch in range(int(epochs)):
        random.shuffle(dataset_training)
        for index in range(len(dataset_training)):
            train = dataset_training[index][0]
            target = dataset_training[index][1]
            Backpropagation.traindata([train], target)
        temp_data_training.append([Backpropagation.do_mse(
            True) * 100, Backpropagation.get_accuracy(True) * 100])
    export_result_train = np.array(temp_data_training)
    export_result_train = pd.DataFrame(
        export_result_train, columns=['MSE', 'ACCURACY'])
    export_result_train.to_csv('report/train-' + output + '.csv')

    for index in range(len(dataset_testing)):
        test = dataset_testing[index][0]
        target = dataset_testing[index][1]

        result = Backpropagation.process([test])
        error_val = abs(target - result)
        error_per = (error_val / target)
        akurasi = (1 - error_per)

        temp_data_result.append([result[0][0], target[0], error_val[0]
                                 [0], error_per[0][0], round(akurasi[0][0] * 100 / 100)])

    export_testing = np.array(temp_data_result)
    export_testing = pd.DataFrame(export_testing, columns=[
                                  'Hasil', 'Target', 'Delta Error', 'Error', 'Akurasi'])
    export_testing.to_csv('report/testing-' + output + '.csv')

    Backpropagation.save_model('models/' + output + '.json')
    saveGraph(output, 'accuracy')
    saveGraph(output, 'error')

    return Backpropagation.get_accuracy(True) * 100, Backpropagation.get_loss(True) * 100


def saveGraph(filename, type):
    result = pd.read_csv('report/train-' + filename + '.csv')
    pt.clf()

    if type == 'accuracy':
        pt.plot(result['ACCURACY'])
        pt.xlabel('Akurasi')
        pt.ylabel('Epoh')
        pt.title('Tingkat akurasi selama pelatihan')
        return pt.savefig('static/fig/akurasi-' + filename + '.png')
    else:
        pt.xlabel('Loss')
        pt.ylabel('Epoh')
        pt.plot(result['MSE'])
        pt.title('Tingkat loss selama pelatihan')
        return pt.savefig('static/fig/error-' + filename + '.png')


def predictData(inputs, credentials):
    weights_ih = np.array(credentials['weights_ih'])
    weights_ho = np.array(credentials['weights_ho'])
    bias_h = np.array(credentials['bias_h'])
    bias_o = np.array(credentials['bias_o'])

    inputs = np.array(inputs)
    hidden = np.dot(inputs, weights_ih)
    hidden = hidden + bias_h

    # Aktivasi
    hidden = mapsigmoid(hidden)

    output = np.dot(hidden, weights_ho)
    output = output + bias_o
    output = mapsigmoid(output)

    return output


def sigmoid(x):
    return 1 / (1 + pow(math.e, -x))


def mapsigmoid(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            data[i, j] = sigmoid(value)
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predictView():
    return render_template('predict.html')


@app.route('/train')
def trainView():
    return render_template('train.html')


@app.route('/training', methods=['POST'])
def training():
    dataset = request.files['dataset_source']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'datasets', secure_filename(dataset.filename))
    dataset.save(file_path)

    filepath = file_path
    learning_rate = request.form['learning_rate']
    epoch = request.form['epoch']
    algorithm = request.form['train_type']
    hidden_neuron = request.form['hidden_neuron']
    model_save_as = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    dataset_training, dataset_testing = processDataset(filepath)

    accuracy, loss = processTrain(
        dataset_training, dataset_testing, learning_rate, epoch, hidden_neuron, model_save_as, algorithm)

    result = {'accuracy': accuracy, 'loss': loss, 'filename': model_save_as}
    return jsonify(result)


@app.route('/predicting', methods=['POST'])
def predicting():
    modelfile = request.form['model']
    with open('models/' + modelfile + '.json') as file:
        model = json.load(file)
    inputs = [float(request.form['dewp']), float(request.form['humi']),
              float(request.form['pres']), float(request.form['temp']), float(request.form['cbwd']), float(request.form['iws']), float(request.form['precipitation']), float(request.form['iprec'])]
    credentials = {
        "weights_ih": model['weights_ih'][0],
        "weights_ho": model['weights_ho'][0],
        "bias_h": model['bias_h'][0],
        "bias_o": model['bias_o'][0],
    }

    result = predictData(inputs, credentials)

    data = {
        "inputs": inputs,
        "credentials": credentials,
        "result": result.tolist()
    }

    return jsonify(data)


@app.route('/data')
def dataView():
    return render_template('index.html')


@app.route('/about')
def aboutView():
    return render_template('index.html')


if __name__ == '__main__':
    app.run('0.0.0.0', 8080, debug=True)
