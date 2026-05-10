from dotenv import load_dotenv
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType

# Load API key
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("No OPENAI_API_KEY found. Please set it in your .env file.")
    st.stop()

# Create LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Create the Wikipedia tool
wiki_api = WikipediaAPIWrapper()

tools = [
    Tool(
        name="Wikipedia Search",
        func=wiki_api.run,
        description="Search Wikipedia for general knowledge questions."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit UI
st.title("Wikipedia + GPT Chatbot")
query = st.text_input("Ask me something:")

if st.button("Search"):
    if query:
        response = agent.run(query)
        st.write(response)