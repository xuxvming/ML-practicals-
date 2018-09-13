# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:35:41 2018

@author: xxiu
"""

'''
input >weight> hidden layer2(activation function)> weights>hidden layer 2
(activation function)>weights> output

1.pass data straight through: feed forward nueral network
2.campare output with intended output > cost or loss fucntion ***(how close is the result)
3.optimaization function > minimize the error fucntion (the difference between expected values and actual values of Y)
This is also known as the LOSS. optimiztation aiming to minimize loss 

backpropagation>go backwards and minipulate the weights which is also one of the optimization methods
the simplest way of adjusting the weights is gradiant descent

The function used in backpropagation is called optimize function 

feed forward + bakcpropagation = epoch (1 cycle)

'''

'''
*** about cost function, loss function and objective function***

Loss function is usually a function defined on a data point, prediction and label, and measures the penalty. 
For example:
square loss l(f(xi|θ),yi)=(f(xi|θ)−yi)2, used in linear regression
hinge loss l(f(xi|θ),yi)=max(0,1−f(xi|θ)yi), used in SVM
0/1 loss l(f(xi|θ),yi)=1⟺f(xi|θ)≠yi, used in theoretical analysis and definition of accuracy

Cost function is usually more general. 
It might be a sum of loss functions over your training set plus some model complexity penalty (regularization). 
For example:
Mean Squared Error MSE(θ)=1N∑Ni=1(f(xi|θ)−yi)2
SVM cost function SVM(θ)=∥θ∥2+C∑Ni=1ξi (there are additional constraints connecting ξi with C and with training set)

Objective function is the most general term for any function that you optimize during training. 
For example, a probability of generating training set in maximum likelihood approach is a well defined objective function, 
but it is not a loss function nor cost function (however you could define an equivalent cost function). 
For example:
MLE is a type of objective function (which you maximize)
Divergence between classes can be an objective function but it is barely a cost function, unless you define something artificial, like 1-Divergence, and name it a cost
Long story short, I would say that:

A loss function is a part of a cost function which is a type of an objective function.


'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

mnist = input_data.read_data_sets("/tmp/data/",one_hot = True)
#one_hot means on is on other is off
#supppose we have 10 classes, 0-9
#0 = [1,0,0,0,0,0,0,0,0,0]> if we want 0 to output is also 0, only the 0th element is on

#define hidden layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size =100

#heigh x width
#2ed parameter is the shape 
#x is the data and y is the label of that data
x =tf.placeholder('float',[None,784])
y =tf.placeholder('float')


def neural_network_model(data):
    #let the weights equal to a tensorflow varibale and random normal varible 
    
    ##would be better to use a for loop in case there are 100 hidden layers
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    #input_data*weights + biases 
    #this is linear model, the purpos of bisaes is to aviod when input_data is 0 
    #so that no neurons are ever fired
    
    #build the model, the behavoir of layers
    #matmul is used for multiplying matrix
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    # activation function Re(ctified) L(inear) (U)nit
    #relu function> f(x) = max(0,x)
    # The derivative of ReLU:
    #1 if x > 0
    #2 otherwise
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])
    
    return output

def train_neural_network(x):
    #rememer the output is onehot array
    prediction = neural_network_model(x)    
    
    #cost fucntion
    #this will calculate the difference of prediction and known label
    cost =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    
    #we want to use a optimizer to minize the cost nest step
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    #define epochs we want 
    hm_epochs = 50
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            #calcluate the loss as we go
            epoch_loss = 0;
            #how many times we need to cycle
            for _ in range(int(mnist.train.num_examples/batch_size)):
                
              epoch_x, epoch_y = mnist.train.next_batch(batch_size)
              
              #c is the cost, run the optimizer for the cost function with feed direction in x and y
              _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
    
              epoch_loss += c
              
            print('Epoch', epoch,'completed out of',hm_epochs,'loss',epoch_loss)
              
       #argmax return the max value of 2
       #it's either 0 or 1 so we need to compare with 1 to get the max value
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
       
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    
    
train_neural_network(x)