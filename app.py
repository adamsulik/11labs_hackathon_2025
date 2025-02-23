import streamlit as st
from openai import OpenAI
from components.components import display_chat_messages, animated_text, display_movie_cards
from dotenv import load_dotenv
from eleven.speak import Speaker
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# from agents import SimpleAgent, convert_messages_to_dict
from agents import EnhancedAgent, convert_messages_to_dict
from network_embedding import MovieMatcher


async def main(movie_matcher: MovieMatcher):
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

    if "movie_ids" not in st.session_state:
        st.session_state.movie_ids = []
    
    
    if "messages" not in st.session_state:
        system_prompt_ask = """
            You are a smart movie assistant. Ask user for their favorite movies and suggest similar movies using the tools you have.\
            Movie list is defined as at least one movie title that user provided. If you have at least one user's movie use a tool to find similar movies.\
        """
        st.session_state.messages = [SystemMessage(system_prompt_ask)]
        # front_agent = SimpleAgent(message_history=st.session_state.messages.copy())
        front_agent = EnhancedAgent(message_history=st.session_state.messages.copy())
        welcome_message = front_agent.graph.invoke({'input_message': 'Hi!'})
        welcome_message = welcome_message['messages'][-1].content
        st.session_state.messages.append(AIMessage(welcome_message))
    else:
        # front_agent = SimpleAgent(message_history=st.session_state.messages.copy())
        front_agent = EnhancedAgent(message_history=st.session_state.messages.copy())

    async def display_chat():
        await display_chat_messages(convert_messages_to_dict(st.session_state.messages))

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
    
    if not len(st.session_state.movie_ids) > 0:
        print('Nothing in session state: ', st.session_state.movie_ids)
        await display_chat()
    else:
        print('Spliting columns')
        col1, col2 = st.columns([6, 4])
        with col1:
            await display_chat()
        with col2:
            await display_movie_cards()
    # st.rerun()

if __name__ == "__main__":
    movie_matcher = MovieMatcher()
    asyncio.run(main(movie_matcher))