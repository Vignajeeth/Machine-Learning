# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 21:00:15 2018

@author: vignajeeth
"""

from Human import *
from GSC import *
from Functions import *
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


# Creates the sigma matrix
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

# Gets a single value which is the product of the below variables with the following dimensions.
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)	#1*41
    T = np.dot(BigSigInv,np.transpose(R))# 41*41  41*1 =41*1
    L = np.dot(R,T)# 1*41  41*1 = 1*1
    return L

# Implements the below formula
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

# This matrix is called the design matrix which depends on the number of clusters formed in K means clustering.
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

#  This matrix finds the weights required for predicting the outputs.
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

# Calculates the Root Mean Square Error
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))




maxAcc = 0.0
maxIter = 0
C_Lambda = 1      #   0.03
TrainingPercent = 100
ValidationPercent = 10
TestPercent = 10
#M              #   10
PHI = []
IsSynthetic = False

xaxis=[]
yaxistr=[]
yaxisval=[]
yaxiste=[]

# Initialise all values from main.py
 
M=3

# For Training Set
TrainingTarget = np.array(YCH_train)
TrainingData   = np.array(XCH_train.values)
TrainingData=TrainingData.T
print(TrainingTarget.shape)
print(TrainingData.shape)

# For Validation Set
ValDataAct = np.array(YCH_val)
ValData    = np.array(XCH_val.values)
ValData=ValData.T
print(ValDataAct.shape)
print(ValData.shape)

# For Test Set
TestDataAct = np.array(YCH_test)
TestData = np.array(XCH_test.values)
TestData=TestData.T
print(ValDataAct.shape)
print(ValData.shape)


ErmsArr = []
AccuracyArr = []

# Performs K means Clustering
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
Mu = kmeans.cluster_centers_

# Calculates the values of things already described above.
BigSigma     = GenerateBigSigma(TrainingData, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(TrainingData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)



print(Mu.shape)            #10*41
print(BigSigma.shape)      #41*41
print(TRAINING_PHI.shape)  #55699*10
print(W.shape)             #10
print(VAL_PHI.shape)       #6962*10
print(TEST_PHI.shape)      #6961*10


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W.T)
VAL_TEST_OUT = GetValTest(VAL_PHI,W.T)
TEST_OUT     = GetValTest(TEST_PHI,W.T)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT.ravel(),TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT.ravel(),ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT.ravel(),TestDataAct))



print ('UBITname      = vignajee')
print ('Person Number = 50291357')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = ",M)
print("Lambda = ",C_Lambda)
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))





# Till this point, closed form solutions was coded
# SGD happens below

print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


W=W.T
W_Now        = np.dot(220, W)
La           = 30        #2
learningRate = 0.01
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

# The below loop performs the gradient descent. The range is 200 because I found the errors to saturate.
for i in range(0,200):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# Printing all the values

print ('----------Gradient Descent Solution--------------------')
print ("M = ",M)
print("Lambda = ",C_Lambda)
print("Learning Rate = ",learningRate)
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
