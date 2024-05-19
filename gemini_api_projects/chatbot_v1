import google.generativeai as genai
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_api_key)
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(f"names are {m.name} and temperature is {m.temperature}")

# Initialize Gemini-Pro 
model = genai.GenerativeModel('gemini-pro')

# Add a Gemini Chat history object to Streamlit session state
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history = [])

#Title for ChatBot Application
st.title("ChatBot with Gemini Pro ")

# Display prompt messages history
for message in st.session_state.chat.history:
    with st.chat_message("ai"):
        st.markdown(message.parts[0].text)

prompt = st.chat_input("How can i help you today.. ")

if prompt:
  st.chat_message("user").markdown(prompt)
  #Pass User Prompt/Message to Gemini Model and get response
  response = st.session_state.chat.send_message(prompt)

  with st.chat_message("ai"):
     st.markdown(response.text)
  #st.write(response['answer'])
