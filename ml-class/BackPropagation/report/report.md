# BP证明及上机实践

### 1. 根据授课内容，给出多分类问题的四个反向传播方程。其中损失函数采用交叉熵，激活函数使用ReLU激活函数

<img src="figures/QQ图片20201125204453.png" alt="QQ图片20201125204453" style="zoom: 33%;" />

<img src="figures/QQ图片20201125204530.png" alt="QQ图片20201125204530" style="zoom: 33%;" />

### 2. 实现手写数字识别程序，要求神经网络包含一个隐层，隐层的神经元个数为15

**激活函数ReLU实现，其中的不可导处的导数当作0处理：**

```python
def relu(z):
    """The relu function."""
    return z * (z > 0)

def relu_prime(z):
    """Derivative of the relu function."""
    return 1.0 * (z > 0)

# 网络输出层使用softmax函数
def softmax(x, axis=0):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s
```

**前向传播：**

```python
def feedforward(self, a):
    """Return the output of the network if ``a`` is input."""
    for b, w in zip(self.biases, self.weights):
        # 是否正在计算输出层
        if (is_equal(b, self.biases[-1])):
            a = softmax(np.dot(w, a) + b)
        else:
            a = relu(np.dot(w, a) + b)
    return a
```

**根据推导出的BP1，输出层误差为：**

```python
# 对比激活函数是sigmoid时：
# delta = self.cost_derivative(activations[-1], y) * relu_prime(zs[-1])
delta = self.cost_derivative(activations[-1], y)
```

**主函数**:

```python
def main():
    training_data, validation_data, test_data = load_data_wrapper()
    network = Network([784, 15, 10])
    network.SGD(training_data=training_data, epochs=30, mini_batch_size=64, eta=3, test_data=test_data)
```

**当激活函数为sigmoid的，参数设置和运行结果为：**

```python
epochs=30, eta=3
```

![sigmoid_result](figures/sigmoid_result.png)

**当激活函数为ReLU，参数设置和运行结果为;**

```python
epochs=30, eta=0.5
```

![relu](figures/relu.png)

经过适当的参数调整，激活函数Relu的效果可与sigmoid相当甚至略优。



### 附录-network完整代码

```python
"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            if (is_equal(b, self.biases[-1])):
                a = softmax(np.dot(w, a)+b)
            else:
                a = relu(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # print("nabla_b:", nabla_b)
        # print("nabla_w:", nabla_w)
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            if (is_equal(b, self.biases[-1])):
                activation = softmax(z)
            else:
                activation = relu(z)
            activations.append(activation)
        # backward pass
        # delta = self.cost_derivative(activations[-1], y) * relu_prime(zs[-1])
        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = relu_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    return z * (z > 0)

def relu_prime(z):
    return 1.0 * (z > 0)

def softmax(x, axis=0):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

def is_equal(a, b):
    res = (a==b)
    if(type(res)==bool):
        return res
    else:
        return all(res)

```



