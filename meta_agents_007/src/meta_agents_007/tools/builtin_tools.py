from crewai_tools import (
    WebsiteSearchTool,
    DirectorySearchTool,
    SerpApiGoogleSearchTool,
    RagTool
)
from dotenv import load_dotenv
import os
load_dotenv()

llm_model = os.getenv('LOCAL_MODEL')
emb_model = os.getenv('EMB_MODEL')
base_url = os.getenv('BASE_URL')
api_key = os.getenv('API_KEY')

LLM_CONFIG = dict(
    model = llm_model,
    api_key = api_key,
    base_url = base_url,
    temperature=1.0,
    max_tokens=8192
)

EMB_CONFIG = dict(
    model = emb_model,
    api_key = api_key,
    api_base = base_url
)

config = dict(
    llm= dict(
        provider='openai',
        config=LLM_CONFIG
    ),
    embedder= dict(
        provider='openai',
        config=EMB_CONFIG
    ),
)


def dir_rag_tool(directory):
    return DirectorySearchTool(directory=directory, config=config)

def web_search_tool(url):
    return WebsiteSearchTool(url=url, config=config)

def rag_tool():
    return RagTool(config=config)

def serapi_google_search_tool():
    return SerpApiGoogleSearchTool()
