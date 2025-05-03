from langchain_openai import ChatOpenAI

def LMStudio(base_url="http://localhost:1234/v1"):
    """
    Creates a connection to a local LLM running via LM Studio API server
    
    Args:
        base_url: URL of the LM Studio API server (default: http://localhost:1234/v1)
    
    Returns:
        A ChatOpenAI instance connected to the local model
    """
    return ChatOpenAI(
        base_url=base_url,
        api_key="not-needed",  # LM Studio doesn't require an API key
    )