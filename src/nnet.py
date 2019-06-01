# coding: UTF-8

import numpy as np
import cPickle
import gzip
import random
import matplotlib.pyplot as plt
from copy import deepcopy

def relu(z):
    return np.maximum(z, 0)
    
def relu_prime(z):
    return np.heaviside(z, 0)

def sigmoid(z):
    sigmoid_range = 34.538776394910684
    z = np.clip(z, -sigmoid_range, sigmoid_range)
    return 1.0 / (1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))
    
def softmax(z):
    #e = np.exp(z)
    a = np.max(z)
    e = np.exp(z - a)
    return  e / np.sum(e)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def random_noise(data, d):
    noised_data = deepcopy(data)
    for j in xrange(len(data)):
        for i in xrange(784):
            if d > random.uniform(0,100):
                noised_data[j][0][i] = random.random()
    return noised_data



f = gzip.open('mnist.pkl.gz', 'rb')
tr_d, va_d, te_d = cPickle.load(f)

training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]] #画素情報を1次元配列に変換
training_results = [vectorized_result(y) for y in tr_d[1]] #正解nに対し10*1の配列のn番目の要素を1にする
training_data = zip(training_inputs, training_results) #[画素情報, 正解]を並べる

test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
test_data = zip(test_inputs, te_d[1])


#training_data = random_noise(training_data, 20) #訓練データにノイズを入れる場合


class Network(object):

    def __init__(self, sizes):
	self.num_layers = len(sizes)
	self.sizes = sizes
	self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
	self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def SGD(self, training_data, epoches, mini_batch_size, eta, noise, test_data=None):
	noised_test_data = self.random_noise(test_data, noise)
	max_rate = 0
	n_test = len(test_data)
        n = len(training_data)
	for j in xrange(epoches):
	    shufful_training_data = np.random.permutation(training_data) #training_dataをシャッフル
	    mini_batches = [shufful_training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)] #shufful_training_dataをミニバッチに分割
	    for mini_batch in mini_batches:
	        self.update_mini_batch(mini_batch, eta)
	    eva = self.evaluate(noised_test_data)
	    print "Epoch {0}: {1} / {2}".format(j, eva, n_test)
	    max_rate = max(max_rate, eva)
	print(max_rate)
	return max_rate
	        

    def random_noise(self, test_data, d):
        noised_test_data = deepcopy(test_data)
        for j in xrange(len(test_data)):
            for i in xrange(self.sizes[0]):
                if d > random.uniform(0,100):
                    noised_test_data[j][0][i] = random.random()
        return noised_test_data
        
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases] #biasesのサイズの要素が0のベクトル
        nabla_w = [np.zeros(w.shape) for w in self.weights] #weightsのサイズの要素が0のベクトル
        for x, y in mini_batch: #x:画素数, y:正解
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #ミニバッチごとにbp学習
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #学習結果をもとにnabla_bを更新
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #学習結果をもとにnabla_wを更新
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)] #nabla_wをもとにbiasesを更新
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)] #nabla_bをもとにweightsを更新

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases] #biasの形の0ベクトル
        nabla_w = [np.zeros(w.shape) for w in self.weights] #weightsの形の0ベクトル
        
        #順伝播
        activation = x
        activations = [] #各層の出力を保存
        zs = [] #各層の入力を保存
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            activations.append(activation)
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z) #隠れ層でsigmoidを使う場合
            #activation = relu(z) #隠れ層でReLUを使う場合
        activations.append(activation)
        z = np.dot(self.weights[-1], activation)+self.biases[-1]
        zs.append(z)
        #activations.append(sigmoid(z)) #出力層でsigmoidを使う場合
        activations.append(softmax(z)) #出力層でsoftmaxを使う場合
        
        #逆伝播
        #delta = (activations[-1]-y)*sigmoid_prime(zs[-1]) #出力層でsigmoidを使う場合
        delta = activations[-1] - y #出力層でsoftmaxを使う場合
        nabla_b[-1] = delta
	nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for j in xrange(2, self.num_layers):
            z = zs[-j]
            pr = sigmoid_prime(z) #隠れ層でsigmoidを使う場合
            #pr = relu_prime(z) #隠れ層でReLUを使う場合
            delta = np.dot(self.weights[-j+1].transpose(), delta) * pr
            nabla_b[-j] = delta
	    nabla_w[-j] = np.dot(delta, activations[-j-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b) #隠れ層でsigmoidを使う場合
            #a = relu(np.dot(w, a)+b) #隠れ層でReLUを使う場合
        return a



def plot_rate(y): 
    x = np.arange(20,150,10)
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
   
    ax.set_ylim(6000,10000)

    ax.plot(x,y[0],label=0)
    ax.plot(x,y[1],label=12.5)
    ax.plot(x,y[2],label=25)
    ax.legend()
    ax.set_xlabel("the number of neurons in a hidden layer")
    ax.set_ylabel("accuracy")
    ax.set_title("4 Layers NN (sigmoid-softmax)")


def nn3(eta):
    res = [[],[],[]]
    for j in xrange(20,150,10): #2層目
        print(j)
        for n in xrange(3):
            print(n*12.5)
            #net = Network([784, j, 10]) #3層の場合 
            net = Network([784, j, j, 10]) #4層の場合
	    print(net.sizes)
            res[n].append(net.SGD(training_data, 10, 10, eta, n*12.5, test_data))
    print(res)
    plot_rate(res)






print("OK")