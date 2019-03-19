
#import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import tensorflow as tf
from GSC import *
from Human import *
import pandas as pd
from keras.models import load_model
from keras.optimizers import Nadam,Adam
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.metrics import accuracy_score
import copy


def sigmoid(arr,k=1):
    ans=1/(1+np.exp(-arr*k))# change derivatives to 10
    return(ans)


def Data_Split(X,Y):    
#    X=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_1.2/Querylevelnorm_X.csv')
#    Y=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_1.2/Querylevelnorm_t.csv')
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=(1/9), random_state=1)
    return(X_train,X_val,X_test,Y_train,Y_val,Y_test)



def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
#    t=0
    accuracy = 0.0
    counter = 0
#    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))



def Ballpark(X_train,X_val,X_test,Y_train,Y_val,Y_test):
#    Data[0],Data[1],Data[2],Data[3]=Human_Dataset() 
#    X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test=Data_Split(Data[i],Data[i+1])
    
    reg = LinearRegression().fit(X_train,Y_train)
    
    preds_train=reg.predict(X_train)
    preds_val=reg.predict(X_val)
    preds_test=reg.predict(X_test)
    
    preds_train=np.asarray(preds_train)
    preds_val=np.asarray(preds_val)
    preds_test=np.asarray(preds_test)
    Y_train=np.asarray(Y_train)
    Y_val=np.asarray(Y_val)
    Y_test=np.asarray(Y_test)
    print()    
    print("Train: ",GetErms(preds_train,Y_train))
    print("Validation: ",GetErms(preds_val,Y_val))
    print("Test: ",GetErms(preds_test,Y_test))
    #73.41663076260232,0.5469740228936869

def Human_Preprocess():
    XCH=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/Created_Data/X_Human_Concat.csv')
    YCH=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/Created_Data/Y_Human_Concat.csv')
    XCH=XCH.drop(['Unnamed: 0'],axis=1)
    #YCH=YCH.drop(['0'],axis=1)
    YCH=YCH.values
    
    XSH=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/Created_Data/X_Human_Subtract.csv')
    YSH=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/Created_Data/Y_Human_Subtract.csv')
    
    XSH=XSH.drop(['Unnamed: 0'],axis=1)
    #YSH=YSH.drop(['0'],axis=1)
    YSH=YSH.values
    XSH=XSH.abs()
    
    #XCH,YCH,XSH,YSH=Human_Dataset()
    #XSH=XSH.abs()
    
    XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test=Data_Split(XCH,YCH)
    XSH_train,XSH_val,XSH_test,YSH_train,YSH_val,YSH_test=Data_Split(XSH,YSH)
    #XCG,YCG,XSG,YSG=GSC_Dataset()
    return(XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,XSH_train,XSH_val,XSH_test,YSH_train,YSH_val,YSH_test)


def GSC_Preprocess():
    XCG=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/Created_Data/X_GSC_Concat_5000.csv')
    YCG=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/Created_Data/Y_GSC_Concat_5000.csv')
    XCG=XCG.drop(['Unnamed: 0'],axis=1)
    YCG=YCG.drop(['0'],axis=1)
    YCG=YCG.values
    YCG=np.append(YCG,0)
    YCG[4999]=1
    YCG=np.reshape(YCG,(10000,1))
    
    
    XSG=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/Created_Data/X_GSC_Subtract_5000.csv')
    YSG=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/Created_Data/Y_GSC_Subtract_5000.csv')
    XSG=XSG.drop(['Unnamed: 0'],axis=1)
    YSG=YSG.drop(['0'],axis=1)
    YSG=YSG.values
    XSG=XSG.abs()
    YSG=np.append(YSG,0)
    YSG[4999]=1
    YSG=np.reshape(YSG,(10000,1))
    
    XCG_train,XCG_val,XCG_test,YCG_train,YCG_val,YCG_test=Data_Split(XCG,YCG)
    XSG_train,XSG_val,XSG_test,YSG_train,YSG_val,YSG_test=Data_Split(XSG,YSG)
#XCG,YCG,XSG,YSG=GSC_Dataset()
    return(XCG_train,XCG_val,XCG_test,YCG_train,YCG_val,YCG_test,XSG_train,XSG_val,XSG_test,YSG_train,YSG_val,YSG_test)


def LinReg(XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,input_no):
    train_len=XCH_train.shape
    input_no=train_len[1]
    x=copy.deepcopy(XCH_train)
    x['b']=1
    w=np.random.rand(input_no+1,1)
    #b=np.random.rand(train_len[0],1)
    y=YCH_train
    
    for i in range(1000):
        y_pred=np.dot(x,w)# none,19 * 19,1=none,1
        
        loss=((y_pred-y)**2)/2
        #delta_y_pred= y_pred-y
        delta_w=np.dot(x.T,(y_pred-y))
        #delta_b=(y_pred-y)
        learnrate=0.00001
        w=w-(learnrate*delta_w)
        #    b=b-(learnrate*delta_b)
    
    xt=copy.deepcopy(XCH_test)
    xt['b']=1
    y_pred_test=np.dot(xt,w)
    print('Accuracy and Loss',GetErms(y_pred_test,YCH_test))    


def LogReg(XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,input_no):
    train_len=XCH_train.shape
    input_no=train_len[1]
    x=copy.deepcopy(XCH_train)
    x['b']=1
    w=(2*(np.random.rand(input_no+1,1))-1)*0.01
    #b=np.random.rand(train_len[0],1)
    y=YCH_train
    
    for i in range(10000):
        a=np.dot(x,w)
        y_pred=sigmoid(a)# none,19 * 19,1=none,1
        
        loss=-((y*np.log(y_pred))+((1-y)*np.log(1-y_pred)))
        
        #delta_y_pred= y_pred-y
        delta_w=np.dot(x.T,-((y/y_pred)+((1-y)/(1-y_pred)))*sigmoid(a)*(1-sigmoid(a)))#1264,1*1264,19
        delta_wr=np.dot(x.T,(y_pred-y))
        #delta_b=(y_pred-y)
        learnrate=0.000000001
        w=w-(learnrate*delta_wr)
    #    b=b-(learnrate*delta_b)
    
    xt=copy.deepcopy(XCH_test)
    xt['b']=1
    y_pred_test=sigmoid(np.dot(xt,w))
    print(((XCH_test.shape[0]-np.sum(np.abs(y_pred_test-YCH_test)))/XCH_test.shape[0])*100)

   

def LinReg_TF(XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,input_no):
#    XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,XSH_train,XSH_val,XSH_test,YSH_train,YSH_val,YSH_test=Human_Preprocess()
    
#    XCH_train['b']=pd.Series(np.ones(1264),index=XCH_train.index)
#    XCH_val['b']=pd.Series(np.ones(159),index=XCH_val.index)
#    XCH_test['b']=pd.Series(np.ones(159),index=XCH_test.index)
    
#    XSH_train['b']=pd.Series(np.ones(1264),index=XSH_train.index)
#    XSH_val['b']=pd.Series(np.ones(159),index=XSH_val.index)
#    XSH_test['b']=pd.Series(np.ones(159),index=XSH_test.index)        
    if input_no==512 or input_no==1024:
        training_epochs=500
    else:
        training_epochs=10000
    
    x=tf.placeholder(tf.float32,[None,input_no])
    w=tf.Variable(tf.random_uniform([input_no,1]))
#    w2=tf.Variable(tf.random_uniform([input_no,1]))
    y=tf.placeholder(tf.float32,[None,1])
    
    
    learning_rate=0.01
#    training_epochs = 5000
    cost_history=np.empty(shape=[1],dtype=float)
    
    init=tf.initialize_all_variables()
    
    t=tf.matmul(x, w)#+tf.matmul(x**2,w2)
    cost=tf.reduce_mean(tf.square(t - y))
    training_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    sess = tf.Session()
    sess.run(init)
    
    for epoch in range(training_epochs):
        sess.run(training_step,feed_dict={x:XCH_train,y:YCH_train})
        cost_history = np.append(cost_history,sess.run(cost,feed_dict={x: XCH_train,y: YCH_train}))
    
    pred_y = sess.run(t, feed_dict={x: XCH_test})
#    mse = tf.reduce_mean(tf.square(pred_y - YCH_test))
#    print("MSE: %.4f" % sess.run(mse)) 
    print('\n\n')    
    print('Accuracy and Loss',GetErms(pred_y,YCH_test))



def LogReg_TF(XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,input_no):
#XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,XSH_train,XSH_val,XSH_test,YSH_train,YSH_val,YSH_test=Human_Preprocess()
#input_no=18    
    #    XCH_train['b']=pd.Series(np.ones(1264),index=XCH_train.index)
    #    XCH_val['b']=pd.Series(np.ones(159),index=XCH_val.index)
    #    XCH_test['b']=pd.Series(np.ones(159),index=XCH_test.index)
    
    #    XSH_train['b']=pd.Series(np.ones(1264),index=XSH_train.index)
    #    XSH_val['b']=pd.Series(np.ones(159),index=XSH_val.index)
    #    XSH_test['b']=pd.Series(np.ones(159),index=XSH_test.index)
    if input_no==512 or input_no==1024:
        training_epochs=500
    else:
        training_epochs=10000
       
    x=tf.placeholder(tf.float32,[None,input_no])
    w=tf.Variable(tf.random_uniform([input_no,1]))
    #    w2=tf.Variable(tf.random_uniform([input_no,1]))
    y=tf.placeholder(tf.float32,[None,1])
    
    
    learning_rate = 0.01
#    training_epochs = 10000
    cost_history = np.empty(shape=[1],dtype=float)
    
    init = tf.initialize_all_variables()
    
    t = tf.sigmoid(tf.matmul(x, w))#+tf.matmul(x**2,w2)
    cost = tf.losses.sigmoid_cross_entropy(y,t)#tf.reduce_mean(tf.square(t - y))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    sess = tf.Session()
    sess.run(init)
    
    for epoch in range(training_epochs):
        sess.run(training_step,feed_dict={x:XCH_train,y:YCH_train})
        cost_history = np.append(cost_history,sess.run(cost,feed_dict={x: XCH_train,y: YCH_train}))
    
    pred_y = sess.run(t, feed_dict={x: XCH_test})
    #    mse = tf.reduce_mean(tf.square(pred_y - YCH_test))
    #    print("MSE: %.4f" % sess.run(mse))
    print('Accuracy   :',accuracy_score(YCH_test,np.around(pred_y))*100)
    #tf.Print('g')
    #print(GetErms(pred_y,YCH_test))



#Concatenation
def GSC_NN(XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,size):
    modelCG=Sequential()
    modelCG.add(Dense(1500,activation='relu',input_shape=(size,)))
    #modelCG.add(Dense(2000,activation='relu'))
    modelCG.add(Dense(2500,activation='relu'))
    #modelCG.add(Dropout(0.3))
    modelCG.add(Dense(1200,activation='relu'))
    modelCG.add(Dense(1,activation='sigmoid'))	
    modelCG.compile(loss='binary_crossentropy',optimizer='Nadam',metrics=['acc'])
    history=modelCG.fit(XCG_train.values,YCG_train,batch_size=64, epochs=100)
    modelCG.save('my_modelCG.h5')        
    
    print("Training Metrics:",modelCG.evaluate(XCG_train.values,YCG_train,verbose=0))    
    print("Validation Metrics:",modelCG.evaluate(XCG_val.values,YCG_val,verbose=0))
    print("Testing Metrics:",modelCG.evaluate(XCG_test.values,YCG_test,verbose=0))



#Concatenation
def Human_Concat_NN(XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test):
    modelC=Sequential()
    
    modelC.add(Dense(50,activation='relu',input_shape=(18,)))
#    modelC.add(Dense(100,activation='relu'))    
#    modelC.add(Dense(70,activation='relu'))
#    modelC.add(Dense(80,activation='relu'))
#    modelC.add(Dropout(0.3))
#    modelC.add(Dense(40,activation='relu'))
    modelC.add(Dense(1,activation='sigmoid'))	
    modelC.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['acc'])    
    history=modelC.fit(XCH_train.values,YCH_train,batch_size=64, epochs=1000)
    modelC.save('my_modelC.h5')        
    print("Loss and Accuracy")
    print("Training Metrics:",modelC.evaluate(XCH_train.values,YCH_train,verbose=0))    
    print("Validation Metrics:",modelC.evaluate(XCH_val.values,YCH_val,verbose=0))
    print("Testing Metrics:",modelC.evaluate(XCH_test.values,YCH_test,verbose=0))


#Subtraction
def Human_Subtract_NN(XSH_train,XSH_val,XSH_test,YSH_train,YSH_val,YSH_test):
    modelS=Sequential()
    
    modelS.add(Dense(50,activation='relu',input_shape=(9,)))
#    modelS.add(Dense(70,activation='relu'))
#    modelS.add(Dense(80,activation='relu'))
#    modelS.add(Dropout(0.3))
#    modelS.add(Dense(40,activation='relu'))
    modelS.add(Dense(1,activation='sigmoid'))	
    modelS.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
    history=modelS.fit(XSH_train.values,YSH_train,batch_size=64, epochs=1000)
    modelS.save('my_modelS.h5')
    print("Loss and Accuracy")
    print("Training Metrics:",modelS.evaluate(XSH_train.values,YSH_train,verbose=0))    
    print("Validation Metrics:",modelS.evaluate(XSH_val.values,YSH_val,verbose=0))
    print("Testing Metrics:",modelS.evaluate(XSH_test.values,YSH_test,verbose=0))
