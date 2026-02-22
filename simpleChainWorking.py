from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

def debug_print(x):
    print("\n==== PROMPT SENT TO MODEL ====\n")
    print(x)
    print("\n==============================\n")
    return x

debug = RunnableLambda(debug_print)

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatOllama(model="llama3.2:1b")

parser = StrOutputParser()

chain = prompt1 | debug |model | parser | prompt2 | debug | model | parser

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)




chain.get_graph().print_ascii()