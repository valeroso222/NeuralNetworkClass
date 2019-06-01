# coding: UTF-8

import cPickle
import gzip
import random
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]] #画素情報を1次元配列に変換
    training_results = [vectorized_result(y) for y in tr_d[1]] #正解nに対し10*1の配列のn番目の要素を1にする
    training_data = zip(training_inputs, training_results) #[画素情報, 正解]を並べる
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e



class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #biasにランダム初期値を入れる
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #weightsにランダム初期値を入れる

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def random_noise(self, data, d):
        for j in xrange(len(data)):
            for i in xrange(self.sizes[0]):
                if d > random.uniform(0,100):
                    data[j][0][i] = random.random()
        return data

    def SGD(self, training_data, epochs, mini_batch_size, eta, noise,
            test_data):
	test_data = self.random_noise(test_data, noise)
        if test_data: n_test = len(test_data)
        n = len(training_data)
	max_rate=0
        for j in xrange(epochs):
            random.shuffle(training_data) #training_dataをシャッフル
            mini_batches = [
		training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)] #training_dataをミニバッチに分割
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
		max_rate = max(max_rate, self.evaluate(test_data))
	print(max_rate)
	return max_rate

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
	for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
	self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb  for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases] #biasの形の0ベクトル
        nabla_w = [np.zeros(w.shape) for w in self.weights] #weightsの形の0ベクトル
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
	nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
	    nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def nn():
    res = [[],[],[],[],[],[]]
    for j in xrange(20,200,1000): #2層目
        for n in xrange(1):
            print(n*5)
            training_data, validation_data, test_data = load_data_wrapper()
            net = Network([784, j, 10])
	    print(net.sizes)
            res[n].append(net.SGD(training_data, 5, 10, 3.0, n*5, test_data=test_data))
    print(res)
    #plot_rate(res)
        
def plot_rate(y):
    x = np.arange(20,200,10)
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
   
    ax.set_ylim(3000,10000)

    ax.plot(x,y[0][:18],label=0)
    ax.plot(x,y[1][:18],label=5)
    ax.plot(x,y[2][:18],label=10)
    ax.plot(x,y[3][:18],label=10)
    ax.plot(x,y[4][:18],label=10)
    ax.plot(x,y[5][:18],label=10)
    ax.legend()
