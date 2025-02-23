from typing import Annotated, List
import streamlit as st
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain.pydantic_v1 import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from network_embedding import MovieMatcher
from typing import List

from dotenv import load_dotenv
load_dotenv()


movie_matcher = MovieMatcher()

def process_suggested_df(sugggested_df):
    movie_ids = sugggested_df['imdb_id'].tolist()
    titles_with_desc = sugggested_df[['title', 'overview']].to_markdown(index=False)
    
    # Update session state only if the movie IDs have changed
    st.session_state.movie_ids = movie_ids
    # st.rerun()    
    return titles_with_desc

class SuggestMoviesSchema(BaseModel):
    query_titles: List[str] = Field(description="List of movie titles to find similar movies for")

@tool(args_schema=SuggestMoviesSchema)
def suggest_movies(query_titles: List[str]) -> str:
    """Based on given movie titles, find similar movies and return a pandas dataframe of recommended movies with their details. You can query all available movies"""
    similar_df = movie_matcher.find_movies(query_titles, threshold=90, n_similar=5)
    return process_suggested_df(similar_df)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    input_message: str

class EnhancedAgent:
    def __init__(self, message_history: List[AnyMessage] = []):
        tools = [suggest_movies]
        self.tools = {tool.name: tool for tool in tools}
        # self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
        self.llm = ChatOpenAI(model='gpt-4o-mini').bind_tools(tools)
        
        self.message_history: List[AnyMessage] = message_history
        self.graph = self.build_graph()
    def build_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node('update_history', self.update_history)
        graph_builder.add_node('llm', self.process_message)
        graph_builder.add_node('tools', self.use_tools)
        graph_builder.add_node('save_history', self.save_history)

        graph_builder.add_edge(START, 'update_history')
        graph_builder.add_edge('update_history', 'llm')
        graph_builder.add_conditional_edges('llm', self.got_action, {True: 'tools', False: 'save_history'})
        graph_builder.add_edge('tools', 'llm')
        graph_builder.add_edge('save_history', END)
        return graph_builder.compile()
    
    def use_tools(self, state: State):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_result = self.tools[tool_name].invoke(tool_args)
            tool_result = ToolMessage(tool_call_id=tool_call['id'], name=tool_name, content=str(tool_result))
            # print(tool_result)
            results.append(tool_result)
        return {'messages': results}


    def update_history(self, state: State):
        initial_message = state.get('input_message', NotImplemented)
        if initial_message is None:
            return {'messages': self.message_history}
        return {'messages': self.message_history + [HumanMessage(initial_message)]}
    
    def process_message(self, state: State):
        response = self.llm.invoke(state['messages'])
        return {'messages': [response]}
    
    def save_history(self, state: State):
        self.message_history = state['messages']

    def got_action(self, state: State):
        return len(state['messages'][-1].tool_calls) > 0
        
    

if __name__ == "__main__":
    sys_message = """
    You are a smart movie assistant. Ask user for their favorite movies and suggest similar movies using the tools you have.\
    Movie list is defined as at least one movie title that user provided. If you have at least one user's movie use a tool to find similar movies.\
    """
    sys_message = [SystemMessage(sys_message)]
    agent = EnhancedAgent(message_history=sys_message)
    while(True):
        message = input('Write a message:')
        if message == 'exit':
            break
        response = agent.graph.invoke({'input_message': message})
        print('AI: ', response['messages'][-1].content)
