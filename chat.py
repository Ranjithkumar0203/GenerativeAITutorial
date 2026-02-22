import os
from langchain_community.chat_models import ChatOllama
from langchain_core.globals import set_debug

set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOllama(model="gemma3:1b-it-qat")

question = input("Enter the question")
response = llm.invoke(question)

print(response.content)