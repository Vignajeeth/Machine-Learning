import pickle
import gzip

from PIL import Image
import os
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import keras
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from tqdm import tqdm
import matplotlib.pyplot as plt

#npa=np.array

def softmax(z):
    z=z-np.max(z)
    ans=(np.exp(z).T/np.sum(np.exp(z),axis=1)).T
    return (ans)

def Data_for_LR(x):
    a=np.ones((x.shape[0],x.shape[1]+1))
    a[:,:-1]=x
    return(a)


def Voting(preds_LR_val_fin,preds1DNNM,preds3SM,preds3RM,M_Val_Y):
    LRVM=keras.utils.to_categorical(preds_LR_val_fin,10)*0.8**2
#    LRVU=keras.utils.to_categorical(preds_LR_U_fin,10)*0.3**2
    DNNVM=keras.utils.to_categorical(preds1DNNM,10)*0.98**2
#    DNNVU=keras.utils.to_categorical(preds1DNNU,10)*0.4**2
    SVMVM=keras.utils.to_categorical(preds3SM,10)*0.94**2
#    SVMVU=keras.utils.to_categorical(preds3SU,10)*0.4**2
    RFVM=keras.utils.to_categorical(preds3RM,10)*.96**2
#    RFVU=keras.utils.to_categorical(preds3RU,10)*.38**2
    
    M_sum=LRVM+DNNVM+SVMVM+RFVM
#    U_sum=LRVU+DNNVU+SVMVU+RFVU
    
    M_Voting_pred=np.argmax(M_sum,axis=1)
#    U_Voting_pred=np.argmax(U_sum,axis=1)
    
    print("Voting Statistics for Test Set")
    print("MNIST Accuracy:   ",accuracy_score(M_Val_Y, M_Voting_pred)*100)
#    print("USPS Accuracy:    ",accuracy_score(U_Y, U_Voting_pred)*100)
    print("Confusion Matrix for MNIST:\n ",confusion_matrix(M_Val_Y, M_Voting_pred))
#    print("Confusion Matrix for USPS:\n ",confusion_matrix(U_Y, U_Voting_pred))


def MNIST():
    filename = 'mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return(training_data, validation_data, test_data)


def USPS():    
    USPSMat  = []
    USPSTar  = []
    curPath  = 'USPSdata/Numerals'
    savedImg = []
    
    for j in range(0,10):
        curFolderPath = curPath + '/' + str(j)
        imgs =  os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg,'r')
                img = img.resize((28, 28))
                savedImg = img
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat.append(imgdata)
                USPSTar.append(j)
    
    USPSMat=np.asarray(USPSMat)
    USPSTar=np.asarray(USPSTar)
    return(USPSMat,USPSTar)



def Random_Forest(M_Train_X, M_Train_Y,U_X,U_Y):
    classifier1R = RandomForestClassifier(n_estimators=3)
    classifier2R = RandomForestClassifier(n_estimators=10)
    classifier3R = RandomForestClassifier(n_estimators=30)
    
    
    classifier1R.fit(M_Train_X, M_Train_Y) #1:46 2:06
    preds1RM=classifier1R.predict(M_Val_X)
    preds1RU=classifier1R.predict(U_X)
    print("\n\nConfiguration 1 (Trees=3)\n")
    print("MNIST Accuracy:   ",accuracy_score(M_Val_Y, preds1RM)*100)
    print("USPS Accuracy:    ",accuracy_score(U_Y, preds1RU)*100)
    print("Confusion Matrix for MNIST:\n ",confusion_matrix(M_Val_Y, preds1RM))
    print("Confusion Matrix for USPS:\n ",confusion_matrix(U_Y, preds1RU))
    
    classifier2R.fit(M_Train_X, M_Train_Y) #1:46 2:06
    preds2RM=classifier2R.predict(M_Val_X)
    preds2RU=classifier2R.predict(U_X)
    print("\n\nConfiguration 2 (Trees=10)\n")
    print("MNIST Accuracy:   ",accuracy_score(M_Val_Y, preds2RM)*100)
    print("USPS Accuracy:    ",accuracy_score(U_Y, preds2RU)*100)
    print("Confusion Matrix for MNIST:\n ",confusion_matrix(M_Val_Y, preds2RM))
    print("Confusion Matrix for USPS:\n ",confusion_matrix(U_Y, preds2RU))
    
    classifier3R.fit(M_Train_X, M_Train_Y) #1:46 2:06
    preds3RM=classifier3R.predict(M_Val_X)
    preds3RU=classifier3R.predict(U_X)
    print("\n\nConfiguration 3 (Trees=30)\n")
    print("MNIST Accuracy:   ",accuracy_score(M_Val_Y, preds3RM)*100)
    print("USPS Accuracy:    ",accuracy_score(U_Y, preds3RU)*100)
    print("Confusion Matrix for MNIST:\n ",confusion_matrix(M_Val_Y, preds3RM))
    print("Confusion Matrix for USPS:\n ",confusion_matrix(U_Y, preds3RU))
    Test_RF=classifier3R.predict(M_Test_X)
    return(Test_RF,preds3RM,preds3RU)


#-------------------SVM-----------------

def SVM(M_Train_X, M_Train_Y,U_X,U_Y):
    classifier1S = SVC(kernel='linear')
    classifier2S = SVC(kernel='rbf', gamma = 1)
    classifier3S = SVC(kernel='rbf')

    classifier1S.fit(M_Train_X, M_Train_Y)
    preds1SM=classifier1S.predict(M_Val_X)
    preds1SU=classifier1S.predict(U_X)
    print("Configuration 1")
    print("MNIST Accuracy:   ",accuracy_score(M_Val_Y, preds1SM)*100)
    print("USPS Accuracy:    ",accuracy_score(U_Y, preds1SU)*100)
    print("Confusion Matrix for MNIST:\n ",confusion_matrix(M_Val_Y, preds1SM))
    print("Confusion Matrix for USPS:\n ",confusion_matrix(U_Y, preds1SU))

    classifier2S.fit(M_Train_X, M_Train_Y)
    preds2SM=classifier2S.predict(M_Val_X)
    preds2SU=classifier2S.predict(U_X)
    print("Configuration 2")
    print("MNIST Accuracy:   ",accuracy_score(M_Val_Y, preds2SM)*100)
    print("USPS Accuracy:    ",accuracy_score(U_Y, preds2SU)*100)
    print("Confusion Matrix for MNIST:\n ",confusion_matrix(M_Val_Y, preds2SM))
    print("Confusion Matrix for USPS:\n ",confusion_matrix(U_Y, preds2SU))

    classifier3S.fit(M_Train_X, M_Train_Y)
    preds3SM=classifier3S.predict(M_Val_X)
    preds3SU=classifier3S.predict(U_X)
    print("Configuration 3")
    print("MNIST Accuracy:   ",accuracy_score(M_Val_Y, preds3SM)*100)
    print("USPS Accuracy:    ",accuracy_score(U_Y, preds3SU)*100)
    print("Confusion Matrix for MNIST:\n ",confusion_matrix(M_Val_Y, preds3SM))
    print("Confusion Matrix for USPS:\n ",confusion_matrix(U_Y, preds3SU))
    Test_SVM=classifier3S.predict(M_Test_X)
    return(Test_SVM,preds3SM,preds3SU)

def DNN(M_Train_X, M_Train_Y,U_X,U_Y):
    X_Tr_DNN=np.reshape(M_Train_X,(50000,28,28,1))
    Y_Tr_DNN=keras.utils.to_categorical(M_Train_Y, 10)
    U_X_DNN=np.reshape(U_X,(19999,28,28,1))
    
    X_Val_DNN=np.reshape(M_Val_X,(10000,28,28,1))
    Y_Val_DNN=keras.utils.to_categorical(M_Val_Y, 10)
    
    model=Sequential()
    model.add(Conv2D(100,(5,5),activation='elu',input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(50,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    history=model.fit(X_Tr_DNN,Y_Tr_DNN,batch_size=1000, epochs=5)
    print(model.evaluate(X_Val_DNN,Y_Val_DNN))
    
    preds1DNNM=model.predict_classes(X_Val_DNN)
    preds1DNNU=model.predict_classes(U_X_DNN)
    
    print("Convolutional Neural Networks")
    print("MNIST Accuracy:   ",accuracy_score(M_Val_Y, preds1DNNM)*100)
    print("USPS Accuracy:    ",accuracy_score(U_Y, preds1DNNU)*100)
    print("Confusion Matrix for MNIST:\n ",confusion_matrix(M_Val_Y, preds1DNNM))
    print("Confusion Matrix for USPS:\n ",confusion_matrix(U_Y, preds1DNNU))
    Test_DNN=model.predict_classes(np.reshape(M_Test_X,(10000,28,28,1)))
    return(Test_DNN,preds1DNNM,preds1DNNU)


def Logistic_Regression(M_Train_X, M_Train_Y,U_X,U_Y):
    WLR=np.random.rand(M_Train_X.shape[1]+1,10) #785*10
    X_Tr_LR=Data_for_LR(M_Train_X)
    #    np.ones((M_Train_X.shape[0],M_Train_X.shape[1]+1))
    #    X_Tr_LR[:,:-1]=M_Train_X #50000,785
    
    X_val_LR=Data_for_LR(M_Val_X)
    #    np.ones((M_Val_X.shape[0],M_Val_X.shape[1]+1))
    #    X_val_LR[:,:-1]=M_Val_X
    U_X_LR=Data_for_LR(U_X)
    #    np.ones((U_X.shape[0],U_X.shape[1]+1))
    #    U_X_LR[:,:-1]=U_X
    loss_mat=[]
    del_lr=0.5
    
    for i in tqdm(range(200)):
        preds1LR=softmax(np.dot(X_Tr_LR,WLR))
        loss=-np.sum(np.log(preds1LR)*Y_Tr_DNN)/50000
        loss_mat.append(loss)
        delta_w=np.dot(X_Tr_LR.T,(preds1LR-Y_Tr_DNN))/50000#785,50000 * 50000,10
        WLR=WLR-del_lr*delta_w
    #        del_lr=(1-(i%50==0)*.3)*del_lr
    
    preds_LR_val=softmax(np.dot(X_val_LR,WLR))
    preds_LR_val_fin=np.argmax(preds_LR_val,axis=1)
    preds_LR_U=softmax(np.dot(U_X_LR,WLR))
    preds_LR_U_fin=np.argmax(preds_LR_U,axis=1)
    
    
    print("Logistic Regression")
    print("MNIST Accuracy:   ",accuracy_score(M_Val_Y, preds_LR_val_fin)*100)
    print("USPS Accuracy:    ",accuracy_score(U_Y, preds_LR_U_fin)*100)
    print("Confusion Matrix for MNIST:\n ",confusion_matrix(M_Val_Y, preds_LR_val_fin))
    print("Confusion Matrix for USPS:\n ",confusion_matrix(U_Y, preds_LR_U_fin))
    Test_LR=np.argmax(softmax(np.dot(Data_for_LR(M_Test_X),WLR)),axis=1)
    return(Test_LR,preds_LR_val_fin,preds_LR_U_fin)


M_Train_Data,M_Val_Data,M_Test_Data=MNIST()
U_X,U_Y=USPS()

M_Train_X=M_Train_Data[0]
M_Train_Y=M_Train_Data[1]
#M_Train_Y=np.reshape(M_Train_Y,(50000,1))

M_Val_X=M_Val_Data[0]
M_Val_Y=M_Val_Data[1]
#M_Val_Y=np.reshape(M_Val_Y,(10000,1))

M_Test_X=M_Test_Data[0]
M_Test_Y=M_Test_Data[1]
#M_Test_Y=np.reshape(M_Test_Y,(10000,1))
Y_Tr_DNN=keras.utils.to_categorical(M_Train_Y, 10)

Test_RF,preds3RM,preds3RU=Random_Forest(M_Train_X, M_Train_Y,U_X,U_Y)
Test_SVM,preds3SM,preds3SU=SVM(M_Train_X, M_Train_Y,U_X,U_Y)
Test_DNN,preds1DNNM,preds1DNNU=DNN(M_Train_X, M_Train_Y,U_X,U_Y)
Test_LR,preds_LR_val_fin,preds_LR_U_fin=Logistic_Regression(M_Train_X, M_Train_Y,U_X,U_Y)


Voting(Test_LR,Test_DNN,Test_SVM,Test_RF,M_Test_Y)


#plt.implot(list(range(200)),loss_mat)

#weights=WLR[:-1,:]
#weigh=[]
#for i in range(10):
#    weigh.append(weights[:,i].reshape(28,28))
#
#plt.imshow(weigh[1].T)
#mrandomf=[81.9,88.26,95.05,96.71,97.16,97.47]
#urandomf=[21.5,24.01,32.45,38.34,41.77,42.49]
#xaxis=[1,3,10,30,100,300]
#
#msvm=[94.23,18.24,94.48]
#usvm=[31.28,10.00,40.49]
#
#plt.title('Support Vector Machine Classifier')
#plt.ylabel('Accuracy')
#plt.xlabel('Configurations')
#confname=['Linear Kernel','RBF with gamma=1','RBF']
#temparr=np.array([1,2,3])
#plt.xticks(temparr, confname)
#
#plt.plot(temparr,msvm,label='MNIST')
#plt.plot(temparr,usvm,label='USPS')
#plt.legend(loc='best')
#
#plt.show()
#

#plt.imshow(model.get_weights()[6][:,4].reshape(10,10))





