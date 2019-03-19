# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 01:15:26 2018

@author: vignajeeth
"""

from Human import *
from GSC import *
from Functions import *
import pandas as pd
from keras.models import load_model
from keras.optimizers import Nadam,Adam
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.metrics import accuracy_score
import numpy as np


from Human import *
from GSC import *
from Functions import *

#------------------HUMAN  MAIN-----------------------
XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,XSH_train,XSH_val,XSH_test,YSH_train,YSH_val,YSH_test=Human_Preprocess()

LinReg(XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,18)
LinReg(XSH_train,XSH_val,XSH_test,YSH_train,YSH_val,YSH_test,9)

LogReg(XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test,18)
LogReg(XSH_train,XSH_val,XSH_test,YSH_train,YSH_val,YSH_test,9)

Human_Concat_NN(XCH_train,XCH_val,XCH_test,YCH_train,YCH_val,YCH_test)
Human_Subtract_NN(XSH_train,XSH_val,XSH_test,YSH_train,YSH_val,YSH_test)

#------------------GSC  MAIN-----------------------

XCG_train,XCG_val,XCG_test,YCG_train,YCG_val,YCG_test,XSG_train,XSG_val,XSG_test,YSG_train,YSG_val,YSG_test=GSC_Preprocess()

LinReg(XCG_train,XCG_val,XCG_test,YCG_train,YCG_val,YCG_test,1024)
LinReg(XSG_train,XSG_val,XSG_test,YSG_train,YSG_val,YSG_test,512)

LogReg(XCG_train,XCG_val,XCG_test,YCG_train,YCG_val,YCG_test,1024)
LogReg(XSG_train,XSG_val,XSG_test,YSG_train,YSG_val,YSG_test,512)

GSC_NN(XCG_train,XCG_val,XCG_test,YCG_train,YCG_val,YCG_test,1024)
GSC_NN(XSG_train,XSG_val,XSG_test,YSG_train,YSG_val,YSG_test,512)









