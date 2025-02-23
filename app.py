import streamlit as st
from posthog.ai.openai import OpenAI
from components.components import display_chat_messages, animated_text, display_movie_cards
from dotenv import load_dotenv
from eleven.speak import Speaker
import asyncio
import posthog

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# from agents import SimpleAgent, convert_messages_to_dict
from agents import EnhancedAgent, convert_messages_to_dict
from network_embedding import MovieMatcher


# Initialize PostHog client
posthog.project_api_key = 'phc_t5qvrjFAKWrXwjOetxLAtvopHGsK3FEIzZBPvVv254G'
posthog.host = "https://us.i.posthog.com"
background_client = OpenAI(
    posthog_client=posthog,
)
background_model = 'gpt-4o-mini'
background_chat = background_client.chat.completions.create

user_id = 'test_user_1'

posthog_params = {
    'posthog_distinct_id': user_id
}
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

    if "layout_ready" not in st.session_state:
        st.session_state.layout_ready = False

    # Create the layout containers early
    if st.session_state.movie_ids:
        col1, col2 = st.columns([6, 4])
        chat_container = col1
        movies_container = col2
    else:
        chat_container = st

    async def handle_chat_interaction(question):
        st.session_state.messages.append(HumanMessage(question))
        with chat_container.chat_message("user"):
            st.markdown(question)

        # Get assistant response
        assistant_response = front_agent.graph.invoke({'input_message': question})
        response_content = assistant_response['messages'][-1].content
        st.session_state.messages.append(AIMessage(response_content))
        
        background_messages_buffer = convert_messages_to_dict(st.session_state.messages)
        background_messages_buffer = [m for m in background_messages_buffer if m['role'] != 'system']
        background_messages_buffer.append({
            "role": "system",
            "content": "Your goal is to summarize user experience and his/her profile and interests. You should return only user's sentiment and interests."
        })
        background_response = background_client.chat.completions.create(messages=background_messages_buffer, model=background_model)
        print(background_response.choices[0].message.content)

        # Handle response animation
        if speak:
            await asyncio.gather(
                speaker.speak_async(response_content),
                animated_text(response_content, container=chat_container)
            )
        else:
            await animated_text(response_content, start_delay=0, container=chat_container)

    async def display_chat():
        await display_chat_messages(
            convert_messages_to_dict(st.session_state.messages), 
            container=chat_container
        )

        if question := st.chat_input("Ask about some movie ..."):
            await handle_chat_interaction(question)

    # Display content
    await display_chat()
    
    # Display movie cards if available
    if st.session_state.movie_ids and 'movies_container' in locals():
        with movies_container:
            await display_movie_cards()

if __name__ == "__main__":
    movie_matcher = MovieMatcher()
    asyncio.run(main(movie_matcher))