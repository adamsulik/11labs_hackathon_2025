import streamlit as st
import time
import asyncio
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


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


async def display_chat_messages(messages):
    for message in messages:
        if message.get("role") == "user":
            with st.chat_message(message["role"]):
                st.markdown(message.get("content"))

        elif message.get("role") == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message.get("content"))


def get_link_preview(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Prioritize Twitter meta tags
        title = (
            soup.find('meta', attrs={'property': 'twitter:title'})
            or soup.find('meta', attrs={'property': 'og:title'})
            or soup.title
        )
        title = title.get('content', '') if hasattr(title, 'get') else (title.string if title else urlparse(url).netloc)
        
        description = ''
        image = ''
        
        # Get meta description with priority
        meta_desc = (
            soup.find('meta', attrs={'property': 'twitter:description'})
            or soup.find('meta', attrs={'property': 'og:description'})
            or soup.find('meta', attrs={'name': 'description'})
        )
        if meta_desc:
            description = meta_desc.get('content', '')
            
        # Get preview image with priority
        meta_img = (
            soup.find('meta', attrs={'property': 'twitter:image'})
            or soup.find('meta', attrs={'property': 'og:image'})
            or soup.find('meta', attrs={'name': 'image'})
        )
        if meta_img:
            image = meta_img.get('content', '')
            
        return {
            'title': title,
            'description': description[:400] + '...' if description else 'No description available',
            'image': image,
            'url': url
        }
    except Exception as e:
        return {
            'title': urlparse(url).netloc,
            'description': 'Failed to load preview',
            'image': '',
            'url': url
        }

async def display_movie_cards():
    for link in [f'https://www.imdb.com/title/{movie_id}' for movie_id in st.session_state.movie_ids]:
        preview = get_link_preview(link)
        
        with st.container():
            st.markdown("---")
            card_html = f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 10px 0; max-width: 100%;">
                    {f'<img src="{preview["image"]}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 4px; margin-bottom: 12px;">' if preview['image'] else ''}
                    <div>
                        <h4 style="margin: 0 0 8px 0; font-size: 16px; font-weight: 600;">{preview['title']}</h4>
                        <p style="color: #666; margin: 0; font-size: 14px;">{preview['description']}</p>
                        <a href="{preview['url']}" target="_blank" style="color: #1E88E5; text-decoration: none; display: block; margin-top: 8px; font-size: 14px;">Visit link →</a>
                    </div>
                </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
