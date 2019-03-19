# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 04:12:22 2018

@author: vignajeeth
"""

import pandas as pd

def GSC_Dataset():
    GSC_Features=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/GSC-Features-Data/GSC-Features.csv')
    Same_Pairs=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/GSC-Features-Data/same_pairs.csv')
    Diff_Pairs=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/GSC-Features-Data/diffn_pairs.csv')
    
    
    A1=pd.merge(Same_Pairs,GSC_Features,left_on='img_id_A',right_on='img_id')
    A2=pd.merge(A1,GSC_Features,left_on='img_id_B',right_on='img_id')
    
    buff1=Diff_Pairs.sample(n=71531)
    A3=pd.merge(buff1,GSC_Features,left_on='img_id_A',right_on='img_id')																																			
    A4=pd.merge(A3,GSC_Features,left_on='img_id_B',right_on='img_id')
    
    Clean1=A2.drop(['img_id_A','img_id_B','img_id_y','img_id_x'],axis=1)
    Clean0=A4.drop(['img_id_A','img_id_B','img_id_y','img_id_x'],axis=1)
    
    Dataset_GSC_Concat=pd.concat([Clean1,Clean0])
    Dataset_GSC_Concat.to_csv('Dataset_GSC_Concat.csv')
    
    Dataset_GSC_Concat=pd.read_csv('/home/vignajeeth/python/Graduate_Codes/ML/Project_2/Dataset_GSC_Concat.csv')
    YC=Dataset_GSC_Concat.target
    XC=Dataset_GSC_Concat.drop(['target'],axis=1)
    
    YC.to_csv('Y_GSC_Concat.csv')
    XC.to_csv('X_GSC_Concat.csv')
    
    
    
    B1=pd.merge(Same_Pairs,GSC_Features,left_on='img_id_A',right_on='img_id')
    B2=pd.merge(Same_Pairs,GSC_Features,left_on='img_id_B',right_on='img_id')
    
    Y1=B1.target
    
    B3=B1.drop(['img_id_A','img_id_B','target','img_id'],axis=1)
    B4=B2.drop(['img_id_A','img_id_B','target','img_id'],axis=1)
    
    X1=B3-B4
    buff1=Diff_Pairs.sample(n=71531)
    
    B5=pd.merge(buff1,GSC_Features,left_on='img_id_A',right_on='img_id')
    B6=pd.merge(buff1,GSC_Features,left_on='img_id_B',right_on='img_id')
    
    Y2=B5.target
    
    B7=B5.drop(['img_id_A','img_id_B','target','img_id'],axis=1)
    B8=B6.drop(['img_id_A','img_id_B','target','img_id'],axis=1)
    
    X2=B7-B8
    
    Y=pd.concat([Y1,Y2])
    X=pd.concat([X1,X2])
    
    Y.to_csv('Y_GSC_Subtract.csv')    # Could have optimised using XC and YC
    X.to_csv('X_GSC_Subtract.csv')
    
    return(XC,YC,X,Y)