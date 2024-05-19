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


chat = model.start_chat(history=[])
while True:
  question = input("Enter your question .... ")
  if question=="Done":
    break
  else:
    resp = chat.send_message(question)
    print(resp.text)

