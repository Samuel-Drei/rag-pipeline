import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import streamlit as st

load_dotenv(find_dotenv())  # sobe nos diretórios pais até achar o .env

def _get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY")


def invoke_ai(system_message: str, user_message: str) -> str:
    """
    Generic function to invoke an AI model given a system and user message.
    Replace this if you want to use a different AI model.
    """

    client = OpenAI(api_key=_get_api_key())  # Insert the API key here, or use env variable $OPENAI_API_KEY.
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content