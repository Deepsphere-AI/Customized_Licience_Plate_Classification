import streamlit as st
from PIL import Image
from Packages.classification import LC_Classification
from Packages.profanity import profanity_check
from Packages.authenticate import authenticate_check

st.set_page_config(layout='wide')

with open('style.css') as f:
	st.markdown (f"<style>{f.read()}</style>",unsafe_allow_html=True)
	


def Reset_fun():
    st.session_state['key1']="Select Option"
    st.session_state['key2']="Library Used"
    st.session_state['key3']="Model Implemented"

Option = st.sidebar.selectbox("",['Select Option','Licence Plate Classification','Profanity Check','Athentication'],key='key1')
st.sidebar.selectbox("",['Library Used','Streamlit','Pandas','Opencv','Tensorflow','easyocr','Openai'],key='key2')
st.sidebar.selectbox("",['Model Implemented','Random Forest','Logistic Regression','Deep Learning','Chat Gpt'],key='key3')
c61,c62,c63 = st.sidebar.columns((1,1,1))
with c61:
	pass
with c62:
	st.sidebar.button("Clear/Reset",on_click=Reset_fun)
with c63:
	pass

# def temp_reset():
# 	st.session_state['key4']="Select to test uploaded Images"




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


if Option == "Licence Plate Classification":
    LC_Classification()

elif Option == "Profanity Check":
	profanity_check()

elif Option == "Athentication":
	authenticate_check()
	