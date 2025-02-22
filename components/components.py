import streamlit as st
import time
import asyncio


def animated_text(text: str, delay: float = 0.02):
    """
    Displays text on the Streamlit app with a typewriter-like animation.

    Parameters:
    - text (str): The full text to display.
    - delay (float): Delay in seconds between displaying each letter.
    """
    # Create a placeholder that will be updated
    placeholder = st.empty()
    displayed_text = ""

    # Loop through each letter and update the placeholder
    for letter in text:
        displayed_text += letter
        placeholder.markdown(f"{displayed_text}")  # Use markdown for styling if needed
        time.sleep(delay)


def display_chat_messages(messages):
    for message in messages:
        if message.get("role") == "user":
            with st.chat_message(message["role"]):
                st.markdown(message.get("content"))

        elif message.get("role") == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message.get("content"))
