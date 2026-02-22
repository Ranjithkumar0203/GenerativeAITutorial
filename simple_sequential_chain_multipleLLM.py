import os
from langchain_community.chat_models import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
titleLlm=ChatOllama(model="llama3.2:1b")
speechLlm=ChatOllama(model="gemma3:1b-it-qat")
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer.
    You need to craft an impactful title for a speech 
    on the following topic: {topic}
    Answer exactly with one title.	
    """
)

speech_prompt = PromptTemplate(
    input_variables=["title","emotion"],
    template="""You need to write a powerful speech of 350 words
     for the following title: {title} with {emotion}
    """
)

first_chain = title_prompt | titleLlm | StrOutputParser() | (lambda title: (st.write(title),title)[1])

second_chain = speech_prompt | speechLlm
final_chain = first_chain |(lambda title:{"title":title, "emotion":emotion} ) |second_chain

final_chain.get_graph().print_ascii()

st.title("Speech Generator")

topic = st.text_input("Enter the topic:")
emotion = st.text_input("Enter the emotion:")


if topic:
    response = final_chain.invoke({"topic":topic})
    st.write(response.content)