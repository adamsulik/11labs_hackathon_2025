import streamlit as st
from openai import OpenAI
from components.components import display_chat_messages, animated_text
from dotenv import load_dotenv
from eleven.speak import Speaker
import asyncio
from tools import movie_tools, process_suggested_df
from network_embedding import MovieMatcher

movie_matcher = MovieMatcher()
suggest_movies = lambda query_titles: process_suggested_df(movie_matcher.find_movies(query_titles, threshold=0.9, n_similar=5))

tool_dictionary = {
    "suggest_movies": suggest_movies 
}


async def main():
    load_dotenv()
    
    with open("prompts/01-system-prompt.txt", "r") as file:
        system_prompt = file.read()
    
    client = OpenAI()
    model = "gpt-4o-mini"
    
    speaker = Speaker()

    st.title("Recomatic 🎥")
    speak = st.toggle("Speaker", value=False)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        st.session_state.messages.append(
            {
                "role": 'system',
                "content": system_prompt
            }
        )
        
        welcome_message = {
            "role": "assistant",
            "content": "Hi, I'm here to help you choose an interesting movie to watch. Let me know what you like, and I'll be happy to pick something for you!",
        }

        st.session_state.messages.append(welcome_message)

    display_chat_messages(st.session_state.messages)

    if question := st.chat_input("Ask about some movie ..."):
        # User question
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Asistant response
        assistant_response = client.chat.completions.create(
            messages=st.session_state.messages,
            model=model,
            tools=movie_tools
        )

        if assistant_response.choices[0].message.tool_calls:
            for tool_call in assistant_response.choices[0].message.tool_calls:
                tool_name = tool_call['function']['name']
                if tool_name == "suggest_movies":
                    movie_ids, data = tool_dictionary[tool_name] # _ is a placeholder for movie_ids
                    st.session_state.messages.append({"role": "assistant", "content": data})
                    assistant_response = client.chat.completions.create(
                        messages=st.session_state.messages,
                        model=model
                    )
        else:
            print('dupa')

        response_content = assistant_response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": response_content})


        if speak:
            await asyncio.gather(
                speaker.speak_async(response_content),
                animated_text(response_content)
            )
        else:
            await animated_text(response_content, start_delay=0)


if __name__ == "__main__":
    asyncio.run(main())