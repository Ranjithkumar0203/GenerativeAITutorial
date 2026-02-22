import os
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import PromptTemplate

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOllama(model="llama3.2:1b",)
prompt_template = PromptTemplate(
    input_variables=["city","month","language","budget"],
    template="""Welcome to the {city} travel guide! 
    If you're visiting in {month}, here's what you can do: 
    1. Must-visit attractions. 
    2. Local cuisine you must try. 
    3. Useful phrases in {language}. 
    4. Tips for traveling on a {budget} budget. 
    Enjoy your trip!
    """
)



city = input("Enter the city:")


print(city)
month = input("Enter the month:")
print(city, month)
language = input("Enter the language:")
print(city, month, language)
budget = input("Enter the budget:")
print(city, month, language, budget)

if city:
    response = llm.invoke(prompt_template.format(city=city,
                                                 month =  month,                                                 
                                                 language=language,
                                                 budget = budget
                                                 ))
    print(response.content)