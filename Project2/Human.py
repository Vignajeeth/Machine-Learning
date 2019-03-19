# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:42:47 2018

@author: vignajeeth
"""

import pandas as pd
#import math
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#import numpy as np



def Human_Dataset():
    Human_Features=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/HumanObserved-Features-Data/HumanObserved-Features-Data.csv')
    Same_Pairs=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/HumanObserved-Features-Data/same_pairs.csv')
    Diff_Pairs=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/HumanObserved-Features-Data/diffn_pairs.csv')
    
    
    A1=pd.merge(Same_Pairs,Human_Features,left_on='img_id_A',right_on='img_id')
    A2=pd.merge(A1,Human_Features,left_on='img_id_B',right_on='img_id')
    
    buff1=Diff_Pairs.sample(n=791)
    A3=pd.merge(buff1,Human_Features,left_on='img_id_A',right_on='img_id')
    A4=pd.merge(A3,Human_Features,left_on='img_id_B',right_on='img_id')
    
    Clean1=A2.drop(['img_id_A','img_id_B','Unnamed: 0_x','img_id_y','Unnamed: 0_y','img_id_x'],axis=1)
    Clean0=A4.drop(['img_id_A','img_id_B','Unnamed: 0_x','img_id_y','Unnamed: 0_y','img_id_x'],axis=1)
    
    
    Dataset_Human_Concat=pd.concat([Clean1,Clean0])
    Dataset_Human_Concat.to_csv('Dataset_Human_Concat.csv')
    
    YC=Dataset_Human_Concat.target
    XC=Dataset_Human_Concat.drop(['target'],axis=1)
    
    YC.to_csv('Y_Human_Concat.csv')
    XC.to_csv('X_Human_Concat.csv')
    
    
    
    B1=pd.merge(Same_Pairs,Human_Features,left_on='img_id_A',right_on='img_id')
    B2=pd.merge(Same_Pairs,Human_Features,left_on='img_id_B',right_on='img_id')
    
    Y1=B1.target
    
    B3=B1.drop(['img_id_A','img_id_B','target','Unnamed: 0','img_id'],axis=1)
    B4=B2.drop(['img_id_A','img_id_B','target','Unnamed: 0','img_id'],axis=1)
    
    X1=B3-B4
    
    B5=pd.merge(buff1,Human_Features,left_on='img_id_A',right_on='img_id')
    B6=pd.merge(buff1,Human_Features,left_on='img_id_B',right_on='img_id')
    
    Y2=B5.target
    
    B7=B5.drop(['img_id_A','img_id_B','target','Unnamed: 0','img_id'],axis=1)
    B8=B6.drop(['img_id_A','img_id_B','target','Unnamed: 0','img_id'],axis=1)
    
    X2=B7-B8
    
    Y=pd.concat([Y1,Y2])
    X=pd.concat([X1,X2])
    
    Y.to_csv('Y_Human_Subtract.csv')    # Could have optimised using XC and YC
    X.to_csv('X_Human_Subtract.csv')
    return(XC,YC,X,Y)

