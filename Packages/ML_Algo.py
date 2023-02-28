from sklearn.linear_model import LinearRegression,LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score,classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from Packages.Process import Processing,Image_Preprocessing
import pandas as pd


score = 0

def Score(X_test_tfidf,y_test,model):
	score_list = []
	y_pred = model.predict(X_test_tfidf.toarray())
	score = accuracy_score(y_pred.round(),y_test)
	# score_list.append((score*100).round())
	return (score*100).round()

def Random_Forest(Hp_Value):
	X_train_tfidf,X_test_tfidf,y_train,y_test,tfidf = Processing(Hp_Value)
	RF_Classifier = RandomForestClassifier()
	RF_Classifier.fit(X_train_tfidf.toarray(),y_train)
	score = Score(X_test_tfidf,y_test,RF_Classifier)
	return score,RF_Classifier

def Logistic_Regression(Hp_Value):
	X_train_tfidf,X_test_tfidf,y_train,y_test,_ = Processing(Hp_Value)
	Logistic_Classifer = LogisticRegression()
	Logistic_Classifer.fit(X_train_tfidf.toarray(),y_train)
	score = Score(X_test_tfidf,y_test,Logistic_Classifer)
	return score,Logistic_Classifer


def Deep_Learning(Hp_Value):
	X_train_tfidf,X_test_tfidf,y_train,y_test,_ = Processing(0.1)

	#Building the Dense Neural network , compiling and applying onto the dataframe
	model = Sequential()
	model.add(Dense(120,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(80,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(60,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(40,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(20,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(1,activation='sigmoid'))

	model.compile(loss=['binary_crossentropy'],optimizer='adam',metrics=['accuracy'])
	History = model.fit(X_train_tfidf.toarray(),y_train,validation_split=0.15,batch_size=50,epochs=Hp_Value) #,callbacks=[callback]

	y_pred = model.predict(X_test_tfidf.toarray())
	score = accuracy_score(y_pred.round(),y_test)
	score = (score*100).round()
	return score,model


