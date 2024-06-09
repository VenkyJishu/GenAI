"""
This program allows to us to upload image and prompt on that image using Streamlit application


For execution
"Streamlit run image_analysis.py"
"""

######## Import the required pacakges ############
import streamlit as st
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv()

####### Provide your api key ###################
#api_key="#####"
#genai.configure(api_key=api_key)
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

######## Choose the heading ###############
st.header("Image analytics")

########## Upload a file ##################
uploaded_file=st.file_uploader("Upload an image",type=["png","jpg","jpeg"])

if uploaded_file is not None:
    st.image(Image.open(uploaded_file))

############ What you want ask ####################
prompt=st.text_input("Enter the text")

######### Use genai skill ##########################

if st.button("GET RESPONSE"):
    img=Image.open(uploaded_file)
    model=genai.GenerativeModel("gemini-1.0-pro-vision-latest")
    response=model.generate_content([prompt,img])
    st.markdown(response.text)
