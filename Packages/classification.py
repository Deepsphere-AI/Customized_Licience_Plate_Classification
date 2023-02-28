import streamlit as st
from Packages.ML_Algo import Deep_Learning,Logistic_Regression,Random_Forest
from Packages.Process import Processing,Image_Preprocessing
from Packages.profanity import profanity_check
import easyocr
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import time




# text_list = []
# result_list = []
# dic = {}
# df = pd.DataFrame(dic)




def LC_Classification():

	pro_state = None
	pro_typ = None
	Algorithm_Selected = None
	Preview_button = None
	select4 = None
	Preview_Rows = None
	# Hp_Value = None
	file_uploaded = []
	all_imgs = []
	empty_list = []
	features_list = []
	df_List = []
	score_list = []
	Test_Preview = False
	test_list = []
	Train_button = False
	Downlaod_Button = False


	c11,c12,c13,c14,c15 = st.columns([0.25,1.5,2.75,0.25,1.75])
	with c12:
		# st.write("")
		st.write("")
		st.write("")
		st.markdown("#### **Problem Statement**")
	with c13:
		pro_state = st.selectbox("",['Select the problem Statement','Vehicle License Plate Classification'],key = "key4")
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
		if pro_state in st_list1:
			#st.write("")
			st.write("")
			st.write("")
			st.markdown("#### **Problem Type**")
	with c13:
		if pro_state in st_list1:
			pro_typ = st.selectbox("",['Select the Problem Type','Classification',])
	with c14:
		st.write("")
	with c15:
		st.write("")


	st_list2 = ['Classification']

	with c11:
		st.write("")
	with c12:
		if pro_typ in st_list2:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("#### **Model Selection**")
	with c13:
		if pro_typ in st_list2:
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

					# Convert the Uploaded file to nd array.

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
			df_List = ["Vehicle Classification Dataset"]
			if select4 in df_List:
				empty_list.append(select4)
			if select4 in df_List:
				df = pd.read_csv("C:/Users/ds_007/anaconda3/envs/VehClass/NumPlate.csv")

	with c24:
		st.write("")
	with c25:
		st.write("")
		st.write("")
		if select4 in df_List:
			Preview_Rows = st.button("Preview 10 Rows")


	with c21:
		st.write("")
	with c22:
		st.write("")
	with c23:
		if Preview_Rows == True and len(empty_list)!=0:
			st.dataframe(df.sample(10),width=700,height=400)
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



		# Hyperparameter for Random Forest

		elif len(empty_list)!=0 and len(features_list)==2 and Algorithm_Selected=="Random Forest":
			Hp_Value = st.slider("Test Size", min_value=0.0, max_value=0.25, value=0.1, step=0.05)


		# Showing Error if all two featrues are not selected

		elif len(features_list)==1 and len(empty_list)!=0:
			st.error("Select the other feature")

	with c34:
		st.write("")
	with c35:
		st.write("")

	with c31:
		st.write("")
	with c32:
		if len(features_list)==2 and Hp_Value!= None:
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


	if  Algorithm_Selected=='Deep Learning' and Train_button==True:

		score,model = Deep_Learning(Hp_Value)
		with c31:
			st.write("")
		with c32:
			st.write("")
		with c33:
			if score != None:
				st.success(f"Training is Sucessfull")

		with c34:
			st.write("")
		with c35:
			st.write("")

	if  Train_button==True and Algorithm_Selected=='Random Forest':
		score,RF_Classifier = Random_Forest(Hp_Value)
		with c31:
			st.write("")
		with c32:
			st.write("")
		with c33:

			if score != None:
				st.success(f"Training is Sucessfull")

		with c34:
			st.write("")
		with c35:
			st.write("")


	if Train_button==True and Algorithm_Selected=='Logistic Regression':

		score,Logistic_Classifer = Logistic_Regression(Hp_Value)
		with c31:
			st.write("")
		with c32:
			st.write("")
		with c33:

			if score != None:
				st.success(f"Training is Sucessfull")

		with c34:
			st.write("")
		with c35:
			st.write("")

	c41,c42,c43,c44,c45 = st.columns([0.25,1.5,2.75,0.25,1.75])

	if len(features_list)==2:
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
			Test_Option =  st.button("Test Uploaded Images")

			if Test_Option is True:
				if Algorithm_Selected == "Logistic Regression":

					dataframe = Image_Preprocessing(all_imgs,Hp_Value,Algorithm_Selected)
					st.dataframe(dataframe,width=600,height=400)
					dataframe.to_csv("C:/Users/ds_007/anaconda3/envs/VehClass/OutPut.csv",index=False)


				elif Algorithm_Selected == "Random Forest":

					dataframe = Image_Preprocessing(all_imgs,Hp_Value,Algorithm_Selected)
					st.dataframe(dataframe,width=600,height=400)
					dataframe.to_csv("C:/Users/ds_007/anaconda3/envs/VehClass/OutPut.csv",index=False)



				elif Algorithm_Selected == "Deep Learning":

					dataframe = Image_Preprocessing(all_imgs,Hp_Value,Algorithm_Selected)
					st.dataframe(dataframe,width=600,height=400)
					dataframe.to_csv("C:/Users/ds_007/anaconda3/envs/VehClass/OutPut.csv",index=False)

		c51,c52,c53 = st.columns([8,4,7])
		with c51:
			st.write("")
		with c52:
			if len(features_list)==2:
				st.write("")
				st.write("")
				st.write("")
				df = pd.read_csv("C:/Users/ds_007/anaconda3/envs/VehClass/OutPut.csv")
				data = df.to_csv().encode('utf-8')
				Downlaod_Button = st.download_button("Download",data,file_name="OutPut.csv",mime='csv')
				Test_Option = "Select to test uploaded Images"

		with c53:
			st.write("")




