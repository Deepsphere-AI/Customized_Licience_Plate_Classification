import spacy
import nltk
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.layers import Dense, Dropout,Flatten
from keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import os
import easyocr
import wordninja
import re
from textblob import Word
from Packages.profanity import profanity_check
from math import sqrt
import openai
import os


st.set_page_config(layout='wide')



with open('style.css') as f:
	st.markdown (f"<style>{f.read()}</style>",unsafe_allow_html=True)

#tensorflow-intel==2.5.0
# tensorflow-estimator==2.8.0
# tensorflow-io-gcs-filesystem==0.30.0
# termcolor==2.2.0

select1 = None
select2 = None
Algorithm_Selected = None
Preview_button = None
Train_button = None
select4 = None
Preview_Rows = None
Hp_Value = None
file_uploaded = []
all_imgs = []
empty_list = []
features_list = []
score_list = []
Test_Preview = False
test_list = []
flag = False
Downlaod_Button = False
# text_list = []
# result_list = []
# dic = {}
# df = pd.DataFrame(dic)




def Reset_fun():
	st.session_state['key1']="Select the problem Statement"
	st.session_state['key2']="Library Used"
	st.session_state['key3']="Model Implemented"
	st.session_state['key4']='Select Option'
	st.session_state['key5']='Library Used'

# def temp_reset():
# 	st.session_state['key4']="Select to test uploaded Images"

def Preprocessing(testsize):
		#Importing the dataframe
	df = pd.read_csv("NumPlate.csv")

	#Resstting the index
	df.reset_index(drop=True,inplace=True)
	#Mapping and converting the catergorical var to Numerical var
	df['Target_Num'] = df['Target'].map({'Personalised':1,'Non_Personalised':0})

    #Dividing  Dependent and Independent Variables  
	Independent_var = df['Detail1'].str.lower()
	Dependent_var = df['Target_Num']


	#Applying the train test split along with startifed cross validation
	x_train,x_test,y_train,y_test = train_test_split(Independent_var, Dependent_var, test_size = testsize, random_state = 225,stratify=df['Target'])

	# Applying the TFIDF Vectorizer
	tfidf = TfidfVectorizer()
	X_train_tfidf = tfidf.fit_transform(x_train)
	X_test_tfidf = tfidf.transform(x_test)
	return X_train_tfidf,X_test_tfidf,y_train,y_test,tfidf

def Score(X_test_tfidf,y_test,model):
	y_pred = model.predict(X_test_tfidf.toarray())
	score = accuracy_score(y_pred.round(),y_test)
	score_list.append((score*100).round())
	return (score*100).round()

# def Test_button_Plot(History):

# 	fig,ax = plt.subplots(1,2,figsize=(15,7))
# 	# summarize history for accuracy
# 	ax[0].plot(History.history['accuracy'])
# 	ax[0].plot(History.history['val_accuracy'])
# 	ax[0].set_title("Accuracy of Data")
# 	ax[0].set_ylabel('Accuracy')
# 	ax[0].set_xlabel('Epochs')
# 	ax[0].legend(['Train', 'Validation'], loc='upper left')
# 	ax[0].set_xticks([])
# 	ax[0].set_yticks([])

# 	# summarize history for loss
# 	ax[1].plot(History.history['loss'])
# 	ax[1].plot(History.history['val_loss'])
# 	ax[1].set_title('Loss of Data')
# 	ax[1].set_ylabel('Loss')
# 	ax[1].set_xlabel('Epochs')
# 	ax[1].legend(['Train', 'Validation'])
# 	ax[1].set_xticks([])
# 	ax[1].set_yticks([])
# 	fig.savefig("plot.png")

def Deep_Learning():
	X_train_tfidf,X_test_tfidf,y_train,y_test,_ = Preprocessing(0.1)

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
	History = model.fit(X_train_tfidf.toarray(),y_train,validation_split=0.25,batch_size=20,epochs=Hp_Value) #,callbacks=[callback]

	score = Score(X_test_tfidf,y_test,model)
	return score,model

def Logistic_Regression():

	X_train_tfidf,X_test_tfidf,y_train,y_test,_ = Preprocessing(Hp_Value)
	Logistic_Classifer = LogisticRegression()
	Logistic_Classifer.fit(X_train_tfidf.toarray(),y_train)
	score = Score(X_test_tfidf,y_test,Logistic_Classifer)
	return score,Logistic_Classifer

def Random_Forest():
	X_train_tfidf,X_test_tfidf,y_train,y_test,_ = Preprocessing(Hp_Value)
	RF_Classifier = RandomForestClassifier()
	RF_Classifier.fit(X_train_tfidf.toarray(),y_train)
	score = Score(X_test_tfidf,y_test,RF_Classifier)
	return score,RF_Classifier

def Image_Preprocessing(model):
	X_train_tfidf,X_test_tfidf,y_train,y_test,tfidf = Preprocessing(0.1)
	file = open('Result.txt','w')
	# result = None
	text_list = []
	result_list = []
	reader = easyocr.Reader(['en'])
	first_list = []
	Threshold = 0
	text = []


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


			if result== 0:
				result_list.append("Non_Personalised")
			elif result== 1:
				result_list.append("Personalised")
			else:
				result_list.append("Unknown")

		except:
			continue
	temper_list = []
	for inner_list in text_list:
		to_text = "   ".join(inner_list[:5]).upper()
		temper_list.append(to_text)
	dictionary = {"Text":temper_list,"Class":result_list}
	dataframe = pd.DataFrame(dictionary)
	st.dataframe(dataframe,width=600,height=400)
	# dataframe.to_csv("Output.csv",index=False)
	# df = pd.read_csv("OutPut.csv")
	# data = df.to_csv().encode('utf-8')
	# Downlaod_Button = st.download_button("Download",data,file_name="OutPut.csv",mime='csv')



col1, col2, col3,col4,col5 = st.columns((2,2,7,2,2))
with col1:
	st.write("")
with col2:
	st.write("")
with col3:
	img = Image.open("Deepsphere_image.png")
	st.image(img,use_column_width=True)
with col4:
	st.write("")
with col5:
	st.write("")

st.markdown("<h1 style='text-align: center; color: Black;font-size: 29px;font-family:IBM Plex Sans;'>Learn to Build Industry Standard Data Science Applications</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: Blue;font-size: 29px;font-family:IBM Plex Sans;'>MLOPS Built On Google Cloud and Streamlit</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: black; font-size:25px;font-family:IBM Plex Sans;'><span style='font-weight: bold'>Problem Statement: </span>Classify Vehicle License Plate</p>", unsafe_allow_html=True)
st.markdown("""<hr style="width:100%;height:2px;background-color:gray;border-width:10">""", unsafe_allow_html=True)

Option = st.sidebar.selectbox("",['Select Option','Licence Plate Classification','Profanity Check'],key='key4')




if Option == 'Licence Plate Classification':

	st.sidebar.selectbox("",['Library Used','Streamlit','Pandas','Opencv','scikit-learn','Tensorflow'],key='key2')
	st.sidebar.selectbox("",['Models Implemented','Random Forest','Logistic Regression','Deep Learning'],key='key3')

	c61,c62,c63 = st.sidebar.columns((1,1,1))
	with c61:
		pass
	with c62:
		st.sidebar.button("Clear/Reset",on_click=Reset_fun)
	with c63:
		pass


	c11,c12,c13,c14,c15 = st.columns([0.25,1.5,2.75,0.25,1.75])
	with c12:
		# st.write("")
		st.write("")
		st.write("")
		st.markdown("#### **Problem Statement**")
	with c13:
		select1 = st.selectbox("",['Select the problem Statement','Vehicle License Plate Classification'],key = "key1")
	with c11:
		st.write("")
	with c14:
		st.write("")
	with c15:
		st.write("")

	st_list1 = ['Vehicle License Plate Classification']


	with c11:
		st.write("")
	with c12:
		if select1 in st_list1:
			#st.write("")
			st.write("")
			st.write("")
			st.markdown("#### **Problem Type**")
	with c13:
		if select1 in st_list1:
			select2 = st.selectbox("",['Select the Problem Type','Classification',])
	with c14:
		st.write("")
	with c15:
		st.write("")


	st_list2 = ['Classification']

	with c11:
		st.write("")
	with c12:
		if select2 in st_list2:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("#### **Model Selection**")
	with c13:
		if select2 in st_list2:
			Algorithm_Selected = st.selectbox("",['Select the Model','Random Forest','Logistic Regression','Deep Learning'])
	with c14:
		st.write("")
	with c15:
		st.write("")

	st_list3 = ['Decision Tree','Random Forest','Logistic Regression','Deep Learning']

	with c11:
		st.write("")
	with c12:
		if Algorithm_Selected in st_list3:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("#### **Upload Input Data**")
	with c13:
		if Algorithm_Selected in st_list3:
			file_uploaded = st.file_uploader("Choose a image file", type=["JPG",'JFIF','JPEG','PNG','TIFF',],accept_multiple_files=True)
			if file_uploaded is not None:
				for file in file_uploaded:

					file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
					all_imgs.append(cv2.imdecode(file_bytes, 1))
	with c14:
		st.write("")
	with c15:
		if Algorithm_Selected in st_list3:
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			Preview_button = st.button('Preview')

	if Preview_button is True:
		cd1,cd2,cd3,cd4,cd5 = st.columns((2,2,2,2,2))
		if len(all_imgs) >=5:
			Display_Images= all_imgs[0:5]
			for i in range(len(Display_Images)):
				with cd1:
					st.image(all_imgs[i])
				with cd2:
					st.image(all_imgs[i+1])
				with cd3:
					st.image(all_imgs[i+2])
				with cd4:
					st.image(all_imgs[i+3])
				with cd5:
					st.image(all_imgs[i+4])
					break
		else:
			with c11:
				st.write("")
			with c12:
				st.write("")
			with c13:
				st.error("Upload atleast 5 Images to preview")
			with c14:
				st.write("")
			with c15:
				st.write("")

	c21,c22,c23,c24,c25 = st.columns([0.25,1.5,2.75,0.25,1.75])

	with c21:
		st.write("")
	with c22:
		if len(file_uploaded)>=1:
			st.write("")
			st.write("")
			# st.write("")
			st.markdown("#### **Training Dataset**")
	with c23:
		if len(file_uploaded)>=1:

			select4 = st.selectbox("",["Select the Dataset","Vehicle Classification Dataset"])
			st_list4 = ["Vehicle Classification Dataset"]
			if select4 in st_list4:
				empty_list.append(select4)
			if select4 in st_list4:
				df = pd.read_csv("NumPlate.csv")

	with c24:
		st.write("")
	with c25:
		st.write("")
		st.write("")
		if len(file_uploaded)>=1:
			Preview_Rows = st.button("Preview 10 Rows")


	with c21:
		st.write("")
	with c22:
		st.write("")
	with c23:
		if Preview_Rows == True and len(empty_list)!=0:
			st.dataframe(df.head(10),width=700,height=400)
		elif Preview_Rows == True and len(empty_list)==0:
			st.error("Select Training Dataset")
	with c24:
		st.write("")
	with c25:
		st.write("")

	c31,c32,c33,c34,c35 = st.columns([0.25,1.5,2.75,0.25,1.75])
	with c31:
		st.write("")
	with c32:
		if len(file_uploaded)>=1 and len(empty_list)!=0:
			st.write("")
			st.write("")
			st.write("")
			#st.write("")
			st.markdown("#### **Feature Engineering**")
	with c33:
		if len(file_uploaded)>=1 and len(empty_list)!=0:
			st.write("")
			features = st.multiselect("Image Features",["Licence Plate Text","Class"])
			features_list.extend(features)

	with c34:
		st.write("")
	with c35:
		st.write("")

	with c31:
		st.write("")
	with c32:
		if len(empty_list)!=0 and len(features_list)==2:
			st.write("")
			st.write("")
			#st.write("")
			st.markdown("#### **Hyper Parameter Tunning**")
	with c33:
		# Hyperparameters for Deep Lerning
		if len(empty_list)!=0 and len(features_list)==2 and Algorithm_Selected=="Deep Learning":
			Hp_Value = st.slider("Number of Epochs", min_value=45, max_value=60, value=50, step=1)	# Hp_Value = st.selectbox("",["Select the HyperParameter","H1","H2"])


		# Hyperparameters for Logistic Regression
		elif len(empty_list)!=0 and len(features_list)==2 and Algorithm_Selected=="Logistic Regression":
			Hp_Value = st.slider("Test Size", min_value=0.0, max_value=0.25, value=0.1, step=0.05)



		#Hyperparameter for Random Forest
		elif len(empty_list)!=0 and len(features_list)==2 and Algorithm_Selected=="Random Forest":
			Hp_Value = st.slider("Test Size", min_value=0.0, max_value=0.25, value=0.1, step=0.05)


		#Showing Error if two featrues are not selected
		elif len(features_list)==1 and len(empty_list)!=0:
			st.error("Select the other feature")
	with c34:
		st.write("")
	with c35:
		st.write("")

	with c31:
		st.write("")
	with c32:
		if len(features_list)==2 and Hp_Value!=None:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("#### **Model Engineering**")
	with c33:
		if len(features_list)==2 and Hp_Value!=None:
			st.write("")
			st.write("")
			Train_button = st.button("Train the model")
	with c34:
		st.write("")
	with c35:
		st.write("")


	if Hp_Value!=None and  Algorithm_Selected=='Deep Learning' and Train_button==True:

		score,_ = Deep_Learning()
		with c31:
			st.write("")
		with c32:
			st.write("")
		with c33:
			if score>=70:
				st.success("Training is Successfull")
		with c34:
			st.write("")
		with c35:
			st.write("")


	if Hp_Value!=None and Train_button==True and Algorithm_Selected=='Random Forest':

		score,_ = Random_Forest()
		with c31:
			st.write("")
		with c32:
			st.write("")
		with c33:
			st.success("Training is Successfully")
		with c34:
			st.write("")
		with c35:
			st.write("")


	if Hp_Value!=None and  Train_button==True and Algorithm_Selected=='Logistic Regression':

		score,_ = Logistic_Regression()
		with c31:
			st.write("")
		with c32:
			st.write("")
		with c33:
			st.success("Training is Successfull")
		with c34:
			st.write("")
		with c35:
			st.write("")

	c41,c42,c43,c44,c45 = st.columns([0.25,1.5,2.75,0.25,1.75])

	if Hp_Value!=None:
		with c41:
			st.write("")
		with c42:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("#### **Model Engineering**")
		with c43:
			st.write("")
			st.write("")
			st.write("")
			Test_Option =  st.button("Test Uploaded Images")#st.selectbox("",["Select to test uploaded Images","Test with the Images Uploaded"],key=4)
			# test_list = ["Test with the Images Uploaded"]
			if Test_Option is True:
				if Algorithm_Selected == "Logistic Regression":
					_,Logistic_Classifer = Logistic_Regression()
					Image_Preprocessing(Logistic_Classifer)
					flag = True


				elif Algorithm_Selected == "Random Forest":
					_,RF_Classifier = Random_Forest()
					Image_Preprocessing(RF_Classifier)
					flag = True



				elif Algorithm_Selected == "Deep Learning":
					_,model = Deep_Learning()
					Image_Preprocessing(model)
					flag = True



					# # image = cv2.resize(image_path,(500,500))
					# if Algorithm_Selected == "Logistic Regression":
					# 	Lp_Class = Image_Preprocessing(image_path,Logistic_Classifer)
					# 	st.write(Lp_Class)
					# elif Algorithm_Selected == "Random_Forest":
					# 	Lp_Class = Image_Preprocessing(image_path,RF_Classifier)
					# 	st.write(Lp_Class)
					# elif Algorithm_Selected == "Deep Learning":
					# 	Lp_Class = Image_Preprocessing(image_path,model)
					# 	st.write(Lp_Class)
				# 	text,result = Image_Preprocessing(image)
				# 	text_list.append(text)
				# 	result_list.append(result)
				# dictionary = {"Licence Plate":text_list,"class":result_list}
				# dataframe = pd.DataFrame(dictionary)
				# st.dataframe(dataframe)
		with c44:
			st.write("")
		with c45:
			st.write("")
			# if Test_Option is True or Test_Option is False:
			# 	st.write("")
			# 	st.write("")
			# 	st.write("")
			# 	Test_Preview = st.button("Preview OutPut")
		# with c41:
		# 	st.write("")
		# with c42:
		# 	st.write("")
		# with c43:
		# 	if Test_Option:
		# 		st.write("")
		# 		df = pd.read_csv("OutPut.csv")
		# 		st.dataframe(df.head(df.shape[0]),width=800,height=400)
		# 	# temp_reset()
		# with c44:
		# 	st.write("")
		# with c45:
		# 	st.write("")

		c51,c52,c53 = st.columns([8,4,7])
		with c51:
			st.write("")
		with c52:
			if flag==True:
				st.write("")
				st.write("")
				st.write("")
				# def convert_df(df):
				# 	return df.to_csv().encode('utf-8')
				# 	df = pd.read_csv("OutPut.csv")
				# csv = convert_df(df)
				df = pd.read_csv("OutPut.csv")
				data = df.to_csv().encode('utf-8')
				Downlaod_Button = st.download_button("Download",data,file_name="OutPut.csv",mime='csv')
				Test_Option = "Select to test uploaded Images"
	# 			if Downlaod_Button == True:
	# 				st.download_button(label="",data=csv,file_name='Model_Outcome.csv',mime='text/csv',
	# )
		with c53:
			st.write("")

elif Option == 'Profanity Check':

	st.sidebar.selectbox("",['Library Used','Streamlit','Pandas','easyocr','opencv'],key='key5')

	c61,c62,c63 = st.sidebar.columns((1,1,1))
	with c61:
		pass
	with c62:
		st.sidebar.button("Clear/Reset",on_click=Reset_fun)
	with c63:
		pass
    
	st_list1 = []
	all_imgs = []
	Preview_button = False
	Check_button = False
	file_uploaded = []
	text_list = []
	response_list = []
	Category = []
	Probability = []
	Reason = []

	def text_extraction(all_imgs):
		confidence_threshold = 0.2
		reader = easyocr.Reader(['en'])
		for image in all_imgs:
			var = " "
			image = cv2.resize(image,(500,300))
			image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			max = 0
			ocr_results = reader.readtext(image)
			for detection in ocr_results:
				if detection[2] > confidence_threshold:
					x0,y0 = [int(value) for value in detection[0][0]]
					x1,y1 = [int(value) for value in detection[0][1]]
					x2,y2 = [int(value) for value in detection[0][2]]
					x3,y3 = [int(value) for value in detection[0][3]]
					dist = (sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
					if dist > max:
						max = dist
						var = detection[1]
			text_list.append(var)
		return text_list

	def word_check(text):
		openai.api_key = os.environ["API_KEY"]
		prompt = "Please provide the probability value and reason for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table for the given word.'"+ text +"'"
		response = openai.Completion.create(
		model="text-davinci-003",
		prompt=prompt,
		temperature=0.82,
		max_tokens=1200,
		top_p=1,
		frequency_penalty=0,
		presence_penalty=0)
		return (response["choices"][0].text)

        

	c11,c12,c13,c14,c15 = st.columns([0.25,1.5,2.75,0.25,1.75])

	with c12:
		# st.write("")
		st.write("")
		st.write("")
		st.markdown("#### **Problem Statement**")
	with c13:
		pb_select = st.selectbox("",['Select the problem Statement','Check for Profanity'],key = "key1")
	with c11:
		st.write("")
	with c14:
		st.write("")
	with c15:
		st.write("")

	st_list1 = ['Check for Profanity']

	with c11:
		st.write("")
	with c12:
		if pb_select  in st_list1:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("#### **Upload Input Data**")
	with c13:
		if pb_select in st_list1:
			file_uploaded = st.file_uploader("Choose a image file", type=["JPG",'JFIF','JPEG','PNG','TIFF',],accept_multiple_files=True)
			if file_uploaded is not None:
				for file in file_uploaded:

					file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
					all_imgs.append(cv2.imdecode(file_bytes, 1))
	with c14:
		st.write("")
	with c15:
		if pb_select  in st_list1 and len(all_imgs)!=0:
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			Preview_button = st.button('Preview')

	if Preview_button is True and len(all_imgs)!=0:
		cd1,cd2,cd3,cd4,cd5 = st.columns((2,2,2,2,2))
		if len(all_imgs) >=5:
			Display_Images= all_imgs[0:5]
			for i in range(len(Display_Images)):
				with cd1:
					st.image(all_imgs[i])
				with cd2:
					st.image(all_imgs[i+1])
				with cd3:
					st.image(all_imgs[i+2])
				with cd4:
					st.image(all_imgs[i+3])
				with cd5:
					st.image(all_imgs[i+4])
					break
		else:
			with c11:
				st.write("")
			with c12:
				st.write("")
			with c13:
				st.error("Upload atleast 5 Images to preview")
			with c14:
				st.write("")
			with c15:
				st.write("")
	c21,c22,c23,c24,c25 = st.columns([0.25,1.5,2.75,0.25,1.75])
	with c22:
		# st.write("")
		st.write("")
		st.write("")
	with c23:
		if pb_select  in st_list1 and len(all_imgs)!=0:
			Check_button = st.button("Check for Profanity")
			# st.markdown(word_check("Bitch"))
			if Check_button == True:
				text_list = text_extraction(all_imgs)
				# st.markdown(text_list)
				for text in text_list:														
					st.success(text) 
					response = word_check(text)
					st.markdown(response)
				# 	st.markdown(response)
				# 	Category.append(response[0])
				# 	Probability.append(response[1])
				# 	Reason.append(response[2])
				# 	response_list.append(word_check(text))
				# dictionary = {"LC_Plate Text":text_list,"Category":Category,"Probability":Probability,"Reason":Reason}
				# dataframe = pd.DataFrame(dictionary)
				# st.dataframe(dataframe,width=600,height=400)
	with c21:
		st.write("")
	with c24:
		st.write("")
	with c25:
		st.write("")


# st.sidebar.selectbox("",['Library Used','Streamlit','Pandas','Opencv','scikit-learn','Tensorflow'],key='key2')
# st.sidebar.selectbox("",['Model Implemented','Random Forest','Logistic Regression','Deep Learning'],key='key3')

# c61,c62,c63 = st.sidebar.columns((1,1,1))
# with c61:
# 	pass
# with c62:
# 	st.sidebar.button("Clear/Reset",on_click=Reset_fun)
# with c63:
# 	pass