from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

def convert_messages_to_dict(messages: List[BaseMessage]) -> List[dict]:
    """
    Convert a list of LangChain messages to a list of dictionaries with role and content.
    
    Args:
        messages: List of LangChain message objects (HumanMessage, AIMessage, SystemMessage)
    
    Returns:
        List of dictionaries with 'role' and 'content' keys
    """
    role_mapping = {
        HumanMessage: 'user',
        AIMessage: 'assistant',
        SystemMessage: 'system'
    }
    
    converted_messages = []
    for message in messages:
        role = role_mapping.get(type(message), 'user')  # default to user if unknown type
        converted_messages.append({
            "role": role,
            "content": message.content
        })
    
    return converted_messages
