import os
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import PromptTemplate

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOllama(model="gemma3:1b-it-qat",)
prompt_template = PromptTemplate(
    input_variables=["country","no_of_paras","language"],
    template="""You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country.
    Avoid giving information about fictional places. If the country is fictional
    or non-existent answer: I don't know.
    Answer the question: What is the traditional cuisine of {country}?
    Answer in {no_of_paras} short paras in {language}
    """
)



country = input("Enter the country:")
while True:
    try:
        no_of_paras = int(input("Enter the number of paras (1-5): "))
        if 1 <= no_of_paras <= 5:
            break
        else:
            print("Please enter a number between 1 and 5.")
    except ValueError:
        print("Invalid input. Please enter a number.")

print(country, no_of_paras)
language = input("Enter the language:")

if country:
    response = llm.invoke(prompt_template.format(country=country,
                                                 no_of_paras=no_of_paras,
                                                 language=language
                                                 ))
    print(response.content)