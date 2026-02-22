import os
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Models
titleLlm = ChatOllama(model="llama3.2:1b")
speechLlm = ChatOllama(model="gemma3:1b-it-qat")

# Prompts
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an experienced speech writer.
    Craft an impactful title for a speech 
    on the following topic: {topic}
    Answer with exactly one title.
    """
)

speech_prompt = PromptTemplate(
    input_variables=["title", "emotion"],
    template="""
    Write a powerful 350-word speech 
    for the following title: {title}
    with a tone of {emotion}.
    """
)

# Chains
title_chain = title_prompt | titleLlm | StrOutputParser()
speech_chain = speech_prompt | speechLlm | StrOutputParser()

# Streamlit UI
st.title("🎤 Speech Generator")

topic = st.text_input("Enter the topic:")
emotion = st.text_input("Enter the emotion:")

# Step 1: Generate Title
if st.button("Generate Title"):
    if topic:
        title = title_chain.invoke({"topic": topic})
        st.session_state.generated_title = title
        st.write("### Generated Title:")
        st.write(title)

# Step 2: Generate Speech (after title exists)
if "generated_title" in st.session_state:
    if st.button("Generate Speech"):
        speech = speech_chain.invoke({
            "title": st.session_state.generated_title,
            "emotion": emotion
        })
        st.write("### Generated Speech:")
        st.write(speech)