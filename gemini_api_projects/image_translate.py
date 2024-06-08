import streamlit as st
from PIL import Image
import google.generativeai as genai
import pyttsx3
from googletrans import Translator
from dotenv import load_dotenv
import os

load_dotenv()

# Streamlit app
st.title("AI Image Description & Translation")
st.write("Upload an image and enter a prompt. The model will generate a description based on your prompt.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
user_prompt = st.text_input("Enter your prompt:", value="")

def generate_audio(i):
    audio = pyttsx3.init()
    audio.setProperty('rate', 150)
    audio.setProperty('volume', 1.0)
    voices = audio.getProperty('voices')
    audio.setProperty('voice', voices[i].id)
    audio.save_to_file(st.session_state.description, "1.mp3")
    audio.runAndWait()

def change_language(language):
    translator = Translator()
    translated_text = translator.translate(st.session_state.description, dest=language)
    return translated_text.text

if uploaded_file and user_prompt:
    try:
        
        if 'description' not in st.session_state:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-1.0-pro-vision-latest')
            st.session_state.img = Image.open(uploaded_file)
            response = model.generate_content([user_prompt, st.session_state.img])
            st.session_state.description = response.text
        st.image(st.session_state.img, caption='Uploaded Image', use_column_width=True)
        st.write(st.session_state.description)
    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    if not uploaded_file:
        st.write("No image file selected.")
    if not user_prompt:
        st.write("Please enter a prompt.")

voice_choice = st.selectbox("Choose a voice:", ["None", "Male", "Female"])
if voice_choice != "None":
    generate_audio(0 if voice_choice == "Male" else 1)
    st.audio("1.mp3", format='audio/mp3')

lang_choice = st.selectbox("Choose a Language to Translate:", ["None", "English", "Hindi", "Odia", "Telugu", "Tamil", "Punjabi", "Malayalam", "Marathi"])
if lang_choice != "None":
    lang_code = {'English': 'en', 'Hindi': 'hi', 'Odia': 'or', 'Telugu': 'te', 'Tamil': 'ta', 'Punjabi': 'pa', 'Malayalam': 'ml', 'Marathi': 'mr'}[lang_choice]
    translated_text = change_language(lang_code)
    st.write(translated_text)
