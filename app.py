import json
from typing import Any
import streamlit as st
from openai import OpenAI
from components.components import display_chat_messages, animated_text
from dotenv import load_dotenv
from eleven.speak import Speaker
import asyncio
from network_embedding import MovieMatcher

movie_matcher = MovieMatcher()

matcher = MovieMatcher()

tools = [{
    "name": "match_movies",
    "type": "function",
    "description": "Find similar movies based on given movie titles",
    "parameters": {
        "type": "object",
        "properties": {
            "movies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of movie titles to find recommendations for",
            }
        },
        "required": ["movies"],
        "additionalProperties": False
        },
    }
]
    
    # Example usage
def match_movies(movies: list[str])->str:    
    # First show matched titles
    similar_movies_df = matcher.find_movies(movies)
    return similar_movies_df.to_string(index=False)


def call_tool(tool_name: str, args: dict[str, Any]) -> dict[str]:
    args: dict[str, Any] = json.loads(args)
    for tool in tools:
        if tool["name"] == tool_name:
            chosen_function = eval(tool_name)
            print(f"Used Function: {chosen_function.__name__}, with {args} parameters")
            return chosen_function(**args)
    raise ValueError(f"Tool: '{tool_name}' not found in tool list")


async def main():
    load_dotenv()
    
    with open("prompts/01-system-prompt.txt", "r") as file:
        system_prompt = file.read()
    with open("prompts/02-system-prompt-ask.txt", "r") as file:
        system_prompt_ask = file.read()
    
    client = OpenAI()
    model = "gpt-4o-mini"
    
    speaker = Speaker()

    st.title("Recomatic 🎥")
    
    col1, col2 = st.columns(2)
    with col1:
        speak = st.toggle("Speaker", value=False)
    with col2:
        if st.button("Restart"):
            st.rerun()
    
    
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        st.session_state.messages.append(
            {
                "role": 'system',
                "content": system_prompt_ask
            }
        )
        
        welcome_response = client.chat.completions.create(
            messages=st.session_state.messages,
            model=model,
            temperature=1,
        )
        
        welcome_message = welcome_response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
        
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
            functions=tools,
            function_call="auto"
        )
        
        print("Response Message: \n",assistant_response.choices[0].message)
        # Check if function call used
        if assistant_response.choices[0].message.function_call:
            function_call = assistant_response.choices[0].message.function_call
            fucntion_name = function_call.name
            function_args = function_call.arguments
            similar_movies = call_tool(fucntion_name, function_args)
            st.session_state.messages.append({"role": "assistant", "content": similar_movies})
            
        else: 
            print("NO tools")
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