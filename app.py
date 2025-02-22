import streamlit as st
from openai import OpenAI
from components.components import display_chat_messages, animated_text
from dotenv import load_dotenv
from eleven.speak import Speaker
import asyncio
from tools import movie_tools, process_suggested_df
from network_embedding import MovieMatcher

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents import SimpleAgent, convert_messages_to_dict

movie_matcher = MovieMatcher()
suggest_movies = lambda query_titles: process_suggested_df(movie_matcher.find_movies(query_titles, threshold=0.9, n_similar=5))

tool_dictionary = {
    "suggest_movies": suggest_movies 
}


async def main():
    load_dotenv()
    with open("prompts/02-system-prompt-ask.txt", "r") as file:
        system_prompt_ask = file.read()
    
    speaker = Speaker()

    st.title("Recomatic 🎥")
    
    col1, col2 = st.columns(2)
    with col1:
        speak = st.toggle("Speaker", value=False)
    with col2:
        if st.button("Restart"):
            st.rerun()
    
    
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(system_prompt_ask)]
        front_agent = SimpleAgent(message_history=st.session_state.messages.copy())
        welcome_message = front_agent.graph.invoke({'input_message': 'Hi!'})
        welcome_message = welcome_message['messages'][-1].content
        st.session_state.messages.append(AIMessage(welcome_message))
    else:
        front_agent = SimpleAgent(message_history=st.session_state.messages.copy())
        
    display_chat_messages(convert_messages_to_dict(st.session_state.messages))


    if question := st.chat_input("Ask about some movie ..."):
        # User question
        st.session_state.messages.append(HumanMessage(question))
        with st.chat_message("user"):
            st.markdown(question)

        # Asistant response
        assistant_response = front_agent.graph.invoke({'input_message': question})
        response_content = assistant_response['messages'][-1].content
        st.session_state.messages.append(AIMessage(response_content))

        if speak:
            await asyncio.gather(
                speaker.speak_async(response_content),
                animated_text(response_content)
            )
        else:
            await animated_text(response_content, start_delay=0)


if __name__ == "__main__":
    asyncio.run(main())