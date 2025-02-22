from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage

from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    input_message: str

class SimpleAgent:
    def __init__(self, message_history: List[AnyMessage] = []):
        self.message_history: List[AnyMessage] = message_history
        self.graph = self.build_graph()
        self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

    def build_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node('update_history', self.update_history)
        graph_builder.add_node('process_message', self.process_message)
        graph_builder.add_node('save_history', self.save_history)
        graph_builder.add_edge(START, 'update_history')
        graph_builder.add_edge('update_history', 'process_message')
        graph_builder.add_edge('process_message', 'save_history')
        graph_builder.add_edge('save_history', END)
        return graph_builder.compile()


    def update_history(self, state: State):
        initial_message = state.get('input_message', NotImplemented)
        if initial_message is None:
            return {'messages': self.message_history}
        return {'messages': self.message_history + [HumanMessage(initial_message)]}
    
    def process_message(self, state: State):
        response = self.llm.invoke(state['messages'])
        return {'messages': AIMessage(response.content)}
    
    def save_history(self, state: State):
        self.message_history = state['messages']
    

if __name__ == "__main__":
    sys_message = [SystemMessage('You are a helpful assistant, who loves animals. Introduce yourself that way if asked.')]
    agent = SimpleAgent(message_history=sys_message)
    while(True):
        message = input('Write a message:')
        if message == 'exit':
            break
        response = agent.graph.invoke({'input_message': message})
        print('AI: ', response['messages'][-1].content)


        