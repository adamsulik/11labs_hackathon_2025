import streamlit as st
import time
import asyncio


async def animated_text(text: str, delay: float = 0.02, start_delay: float = 1.5):
    """
    Displays text on the Streamlit app with a typewriter-like animation.

    Parameters:
    - text (str): The full text to display.
    - delay (float): Delay in seconds between displaying each letter.
    - start_delay (float): Animation start delay due to elevenlabs latency
    """
    # Create a placeholder that will be updated
    with st.chat_message('assistant'):
        placeholder = st.empty()
        displayed_text = ""
        await asyncio.sleep(start_delay)
        
        # Loop through each letter and update the placeholder
        for letter in text:
            displayed_text += letter
            placeholder.markdown(f"{displayed_text}")  # Use markdown for styling if needed
            await asyncio.sleep(delay)


def display_chat_messages(messages):
    for message in messages:
        if message.get("role") == "user":
            with st.chat_message(message["role"]):
                st.markdown(message.get("content"))

        elif message.get("role") == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message.get("content"))
