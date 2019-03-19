#	equal	0
#	fizz	1
#	buzz	2
#	fb	3

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from keras import *
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

from numpy.random import seed
seed(17)
from tensorflow import set_random_seed
set_random_seed(11)

def Dataprep():							# This function is to create the dataset and make it easier to input
	dataset=np.zeros((1000))
	for i in range(1000):
		if (i+1)%15==0:
			dataset[i]=3
		elif (i+1)%5==0:
			dataset[i]=2
		elif (i+1)%3==0:
			dataset[i]=1

	#Y=pd.Series(dataset)
	#print(Y)

	test_data=dataset[:100]
	Y_test=pd.Series(test_data)
	train_data=dataset[100:]
	Y_train=pd.Series(train_data)


	X_test=np.zeros(100)
	X_train=np.zeros(900)

	for i in range(100):
		X_test[i]=i+1
	for i in range(900):
		X_train[i]=i+101

	X_test=X_test.astype(np.int32)
	X_train=X_train.astype(np.int32)
								# The 6 for loops below are to convert int to 10 bit vectors
	X_testfinal=[]
	X_trainfinal=[]
	for i in range(100):
		X_testfinal.append(str(bin(X_test[i]))[2:])

	for i in range(900):
		X_trainfinal.append(str(bin(X_train[i]))[2:])


	for i in range(100):
		X_testfinal[i]='0'*(10-len(X_testfinal[i]))+X_testfinal[i]

	for i in range(900):
		X_trainfinal[i]='0'*(10-len(X_trainfinal[i]))+X_trainfinal[i]
	
	for i in range(100):
		X_testfinal[i]=list(X_testfinal[i])
		for j in range(10):
			X_testfinal[i][j]=int(X_testfinal[i][j])

	for i in range(900):
		X_trainfinal[i]=list(X_trainfinal[i])
		for j in range(10):
			X_trainfinal[i][j]=int(X_trainfinal[i][j])


	#X_test=pd.Series(X_test)
	#X_train=pd.Series(X_train)
	
	Y_test=pd.get_dummies(Y_test)
	Y_train=pd.get_dummies(Y_train)
	Y_test=Y_test.values
	Y_train=Y_train.values
	#print(X_test,Y_test)
	return(X_trainfinal, X_testfinal, Y_train, Y_test)
	
def Datasplit(X_train,Y_train,split=0.25):		# This function is to split the data into Training and Cross Validation
	seed=5
	X_trainf, X_cv, Y_trainf, Y_cv=model_selection.train_test_split(X_train,Y_train, test_size=split, random_state=seed)
	return(X_trainf, X_cv, Y_trainf, Y_cv)




def TrainingNN(X_trainf, X_cv, Y_trainf, Y_cv, epoch=1000,O=0,A=0,D=0):		# This function is where the NeuralNets train
	
	
	optims=[optimizers.SGD(lr=0.4), optimizers.RMSprop(lr=0.4), 'Adagrad', 'Adadelta', 'Adam' ,'Adamax' ,'Nadam']		#7
	
	activs=['linear','sigmoid','tanh','softsign','relu','softplus']	#6
	drops=[0,0.3,0.6,0.9]			#4

	model=Sequential()


	model.add(Dense(50,activation=activs[A],input_dim=10))
	model.add(Dense(70,activation=activs[A]))
	model.add(Dense(80,activation=activs[A]))
	model.add(Dropout(drops[D]))
	model.add(Dense(40,activation=activs[A]))
	model.add(Dense(4,activation='softmax'))	
	
	model.compile(loss='categorical_crossentropy', optimizer=optims[O], metrics=['acc'])

	history=model.fit(X_trainf,Y_trainf,batch_size=64, epochs=epoch)			

	model.save('my_model.h5')

#Train cv split				0.25
#Layers					50 70 80 0.1D 40 4
#Optimiser				Nadam
#Epochs					300
#Activation fn
#LR if not Nadam
#Regularizing
#model = load_model('my_model.h5')

	#mllist=[]
	#mlpredcv=[]
	#mlpredtr=[]
	#mllist.append(RandomForestClassifier())
	#mllist.append(GaussianNB())
	#for i in range(1):
	#	mllist[i].fit(X_trainf,Y_trainf)
	#	mlpredcv.append(mllist[i].predict(X_cv))
	#	mlpredtr.append(mllist[i].predict(X_trainf))
	#for i in range(1):
	#	print("\nRF Training (Should be 100)				:",accuracy_score(Y_trainf, mlpredtr[i])*100)
	#	print("RF Cross Validation (Hopefully 100)		:",accuracy_score(Y_cv, mlpredcv[i])*100)



	predictioncv=model.predict_classes(X_cv)				
	predictrain=model.predict_classes(X_trainf)
	PREDCV=np.argmax(Y_cv,axis=1)
	PREDTR=np.argmax(Y_trainf,axis=1)
	
	cvaccuracy=accuracy_score(PREDCV, predictioncv)*100
	print("\n\nTraining 			:",accuracy_score(PREDTR, predictrain)*100)
	print("Cross Validation 		:",cvaccuracy)
	
	
	
	optimsname=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#	print(history.history.keys())
	plt.plot(history.history['acc'])
#	plt.plot(history.history['val_acc'])
	Title=(optimsname[O]+' Optimiser with '+activs[A]+' Activation having '+str(drops[D])+' Dropout')
	plt.title(Title)
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend([optimsname[O]+' Optimiser with '+activs[A]], loc='upper left')
	plt.show()
#	savefilename='test'+str(O)+str(A)+str(D)+'a.png'
#	plt.savefig(savefilename, bbox_inches='tight')
	
	return(predictioncv,predictrain,cvaccuracy)

def Test(X_test,Y_test):
	model = load_model('my_model.h5')
	TestAns=model.predict_classes(X_test)
	GT=np.argmax(Y_test,axis=1)
	print("\nTestset Accuracy		:",accuracy_score(GT, TestAns)*100)
	
	csvtest=[]
	csvtestf=[]
	for i in range(100):
		csvtest.append(i+1)

	for i in range(100):
		csvtestf.append(str(bin(csvtest[i]))[2:])
	
	for i in range(100):
		csvtestf[i]='0'*(10-len(csvtestf[i]))+csvtestf[i]

	for i in range(100):
		csvtestf[i]=list(csvtestf[i])
		for j in range(10):
			csvtestf[i][j]=int(csvtestf[i][j])

	model = load_model('my_model.h5')
	TestAns=model.predict_classes(csvtestf)


	GRTR=np.zeros((100))
	for i in range(100):
		if (i+1)%15==0:
			GRTR[i]=3
		elif (i+1)%5==0:
			GRTR[i]=2
		elif (i+1)%3==0:
			GRTR[i]=1


	GRTRSTR=[]
	for i in range(100):
		if GRTR[i]==0:
			GRTRSTR.append('Other')
		elif GRTR[i]==1:
			GRTRSTR.append('Fizz')
		elif GRTR[i]==2:
			GRTRSTR.append('Buzz')
		elif GRTR[i]==3:
			GRTRSTR.append('Fizzbuzz')


	testansstr=[]
	for i in range(100):
		if TestAns[i]==0:
			testansstr.append('Other')
		elif TestAns[i]==1:
			testansstr.append('Fizz')
		elif TestAns[i]==2:
			testansstr.append('Buzz')
		elif TestAns[i]==3:
			testansstr.append('Fizzbuzz')
	
	
	csvtest.insert(0,"UBID")
	GRTRSTR.insert(0,"vignajee")
	csvtest.insert(1,"Person Number")
	GRTRSTR.insert(1,"50291357")
	testansstr.insert(0,'')
	testansstr.insert(1,'')
	DFNew={'Testing Inputs':csvtest,'Actual Outputs':GRTRSTR,'Predicted Outputs':testansstr}
	DFNew=pd.DataFrame(DFNew)
	DFNew=DFNew[['Testing Inputs','Actual Outputs','Predicted Outputs']]
	pd.DataFrame(DFNew).to_csv('Output.csv')



#_________________-Main-_____________________

#from fizzbuzz import *
X_train, X_test, Y_train, Y_test = Dataprep()
X_trainf, X_cv, Y_trainf, Y_cv = Datasplit(X_train,Y_train)
#Acmat=[]

#for i in range(6):
#	for j in range(7):
predcv,predtr,CVac = TrainingNN(X_trainf, X_cv, Y_trainf, Y_cv,O=6,A=2,D=1) # The model having Nadam optimiser, tanh function and a droprate of 0.3 was the most accurate in CV set.
Test(X_test,Y_test)		# This function was added after iterating through all possible combinations and finding the best one ans was called only once.

#Acmat.append([CVac,j,i,3])		






