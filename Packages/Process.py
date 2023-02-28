import easyocr
import glob
import re
import pandas as pd
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from math import sqrt
import streamlit as st
import os
import wordninja
import re
from textblob import Word
import random

# from Packages.ML_Algo import Deep_Learning,Random_Forest,Logistic_Regression
import Packages.ML_Algo as Algo

def Processing(testsize):
	#Importing the dataframe

	df = pd.read_csv("C:/Users/ds_007/anaconda3/envs/VehClass/NumPlate.csv")

	# Resstting the index

	df.reset_index(drop=True,inplace=True)

	# Mapping and converting the catergorical var to Numerical var

	df['Target_Num'] = df['Target'].map({'Personalised':1,'Non_Personalised':0})

    # Dividing  Dependent and Independent Variables 
	
	Independent_var = df['Detail1'].str.lower()
	Dependent_var = df['Target_Num']


	# Applying the train test split along with startifed cross validation

	x_train,x_test,y_train,y_test = train_test_split(Independent_var, Dependent_var, test_size = testsize, random_state = 225,stratify=df['Target'])

	# Applying the TFIDF Vectorizer

	tfidf = TfidfVectorizer()
	X_train_tfidf = tfidf.fit_transform(x_train)
	X_test_tfidf = tfidf.transform(x_test)
	return X_train_tfidf,X_test_tfidf,y_train,y_test,tfidf





def Image_Preprocessing(all_imgs,Hp_Value,Algorithm_Selected):
	X_train_tfidf,X_test_tfidf,y_train,y_test,tfidf = Processing(0.1)
	if Algorithm_Selected == "Deep Learning":
		score,model = Algo.Deep_Learning(Hp_Value)
	if Algorithm_Selected == "Logistic Regression":
		score,model = Algo.Logistic_Regression(Hp_Value)
	if Algorithm_Selected == "Random Forest":
		score,model = Algo.Random_Forest(Hp_Value)

	result = None
	text_list = []
	result_list = []
	reader = easyocr.Reader(['en'])
	first_list = []
	Threshold = 0.2
	text = []
	class_list = ["Persnalised","Non Personalised"]


	for image_path in all_imgs:
		try:
			image = cv2.resize(image_path,(400,300))
			ocr_results = reader.readtext(image)

			temp_list = []
			for detection in ocr_results:
				if detection[2]>=Threshold:
					temp_list.append(detection[1])
			text_list.append(temp_list)

			for detection in ocr_results:
				if detection[2] >= Threshold:
					text.append(detection[1])
			first_list.append(text)

			def check_sentence_spelling(words):
				fun_list = []
				new_list = []
				words = [word.upper() for word in words]
				words = [re.sub(r'[^A-Za-z0-9]+', '', word) for word in words]
				for word in words:
					new_list.extend(wordninja.split(word))
				for word in new_list:
					word = Word(word)
					result = word.spellcheck()
					fun_list.append(result[0][0])
				return fun_list



			per_list = []
			for detail in first_list:
				str_frm_fun = check_sentence_spelling(detail)
				per_list.append(str_frm_fun)


			per_duplicate = per_list.copy()

			stop_word_list1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','of'] # 
			stop_word_list2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
								48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 
								94, 95, 96, 97, 98]
			stop_word_list3 = ['01','02','03','04','05','06','07','08','09']

			for index,detail in enumerate(per_duplicate):
				for i in detail:
					if len(i)==1 or len(i)==2:
						per_duplicate[index].remove(i)
				for i in detail:
					if i in stop_word_list1 or i in stop_word_list2 or i in stop_word_list3:
						per_duplicate[index].remove(i)

			text = []
			for detail in per_duplicate:
				text.extend(detail)
			text = " ".join(text)
			text = [text]
			try:
				output = tfidf.transform(text)
				example = output.toarray()

				result = model.predict(example)
				result = round(result[0][0])
			except Exception as e:
				pass



			if result== 1:
				result_list.append("Personalised")
			elif result== 0:
				result_list.append("Non Personalised")
			else:
				result_list.append("UnKnown")
		except:
			continue
		
	temper_list = []
	for inner_list in text_list:
		to_text = "   ".join(inner_list[:5]).upper()
		temper_list.append(to_text)

	dictionary = {"Text":temper_list,"Class":result_list}
	dataframe = pd.DataFrame(dictionary)

	return dataframe


