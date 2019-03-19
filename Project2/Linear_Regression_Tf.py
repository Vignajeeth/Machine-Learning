# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 04:17:26 2018

@author: vignajeeth
"""


from Functions import *
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


def Human_LogReg(XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,input_no):
XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,XSH_train,XSH_val,XSH_test,YSH_train,YSH_val,YSH_test=Human_Preprocess()
input_no=1024    
    #    XCH_train['b']=pd.Series(np.ones(1264),index=XCH_train.index)
    #    XCH_val['b']=pd.Series(np.ones(159),index=XCH_val.index)
    #    XCH_test['b']=pd.Series(np.ones(159),index=XCH_test.index)
    
    #    XSH_train['b']=pd.Series(np.ones(1264),index=XSH_train.index)
    #    XSH_val['b']=pd.Series(np.ones(159),index=XSH_val.index)
    #    XSH_test['b']=pd.Series(np.ones(159),index=XSH_test.index)

x=tf.placeholder(tf.float32,[None,input_no])
w=tf.Variable(tf.random_uniform([input_no,1]))
#    w2=tf.Variable(tf.random_uniform([input_no,1]))
y=tf.placeholder(tf.float32,[None,1])


learning_rate = 0.01
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)

init = tf.initialize_all_variables()

t = tf.sigmoid(tf.matmul(x, w))#+tf.matmul(x**2,w2)
cost = tf.losses.sigmoid_cross_entropy(y,t)#tf.reduce_mean(tf.square(t - y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={x:XCG_train,y:YCG_train})
    cost_history = np.append(cost_history,sess.run(cost,feed_dict={x: XCG_train,y: YCG_train}))

pred_y = sess.run(t, feed_dict={x: XCG_test})
#    mse = tf.reduce_mean(tf.square(pred_y - YCH_test))
#    print("MSE: %.4f" % sess.run(mse))
print('Accuracy   :',accuracy_score(YCG_test,np.around(pred_y))*100)
#tf.Print('g')
#print(GetErms(pred_y,YCH_test))









input_no=512
    x=tf.placeholder(tf.float32,[None,input_no])
    w=tf.Variable(tf.random_uniform([input_no,1]))
    #    w2=tf.Variable(tf.random_uniform([input_no,1]))
    y=tf.placeholder(tf.float32,[None,1])
    
    
    learning_rate = 0.01
    training_epochs = 500
    cost_history = np.empty(shape=[1],dtype=float)
    
    init = tf.initialize_all_variables()
    
    t = tf.matmul(x, w)#+tf.matmul(x**2,w2)
    cost = tf.reduce_mean(tf.square(t - y))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    sess = tf.Session()
    sess.run(init)
    #Takes 13 mins
    for epoch in range(training_epochs):
        sess.run(training_step,feed_dict={x:XSG_train.values,y:YSG_train})
        cost_history = np.append(cost_history,sess.run(cost,feed_dict={x: XSG_train.values,y: YSG_train}))
    
    pred_y = sess.run(t, feed_dict={x: XSG_test})
    #    mse = tf.reduce_mean(tf.square(pred_y - YCH_test))
    #    print("MSE: %.4f" % sess.run(mse)) 
    print('\n')    
    print('Accuracy and Loss',GetErms(pred_y,YSG_test))

