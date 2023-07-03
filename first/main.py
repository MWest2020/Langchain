import os
from dotenv import load_dotenv
import requests
import fnmatch
import argparse
import base64

# needed for the display of costs
from langchain.callbacks import get_openai_callback


with get_openai_callback() as cb:


    load_dotenv()

    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    OPENAPI_KEY = os.getenv("OPENAI_API_KEY")

    from langchain.agents import AgentType, initialize_agent, load_tools
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.llms import OpenAI
    from langchain.docstore.document import Document
    from langchain.chains import RetrievalQA
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains.question_answering import load_qa_chain
    
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    from langchain import OpenAI, ConversationChain


    # the folowing uses a user input form the terminal and uses google search in a chain to find the answer
    # The language model we're going to use to control the agent.
    llm = OpenAI(temperature=0.9)

    # The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


    prompt = input("Enter a prompt: ")
    # Let's test it out!
    agent.run(prompt)
 
    


    print(f"costs: {cb}")
    