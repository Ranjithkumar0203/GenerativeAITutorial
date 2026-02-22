from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

# ------------------ MODELS ------------------
model1 = ChatOllama(model="llama3.2:1b")
model2 = ChatOllama(model="gemma3:1b-it-qat")
model3 = ChatOllama(model="my-service-agent:latest")

# ------------------ PROMPTS ------------------
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short quiz questions from the following text:\n{text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the following notes and quiz into a final structured output:\nNotes:\n{notes}\n\nQuiz:\n{quiz}",
    input_variables=["notes", "quiz"],
)

# ------------------ PARSER ------------------
parser = StrOutputParser()

# ------------------ DEBUG FUNCTION ------------------
def debug_print(x):
    print("\n========== PROMPT SENT TO MODEL ==========\n")
    print(x)
    print("\n==========================================\n")
    return x

debug = RunnableLambda(debug_print)

# ------------------ PARALLEL CHAIN ------------------
parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | debug | model1 | parser,
        "quiz": prompt2 | debug | model2 | parser,
    }
)

# ------------------ MERGE CHAIN ------------------
merge_chain = prompt3 | debug | model3 | parser

# ------------------ FINAL CHAIN ------------------
chain = parallel_chain | merge_chain

# ------------------ INPUT ------------------
text = """
Dependency Injection (DI)
The technology that Spring is most identified with is the Dependency Injection
(DI) flavor of Inversion of Control. The Inversion of Control (IoC) is a general concept,
and it can be expressed in many different ways. Dependency Injection is merely one
concrete example of Inversion of Control.
When writing a complex Java application, application classes should be as independent as
possible of other Java classes to increase the possibility to reuse these classes and to test
them independently of other classes while unit testing. Dependency Injection helps in
gluing these classes together and at the same time keeping them independent.
What is dependency injection exactly? Let's look at these two words separately. Here the
dependency part translates into an association between two classes. For example, class A
is dependent of class B. Now, let's look at the second part, injection. All this means is,
class B will get injected into class A by the IoC.
Dependency injection can happen in the way of passing parameters to the constructor or
by post-construction using setter methods. As Dependency Injection is the heart of Spring
Framework, we will explain this concept in a separate chapter with relevant example.
Aspect Oriented Programming (AOP)
One of the key components of Spring is the Aspect Oriented Programming
(AOP) framework. The functions that span multiple points of an application are
called cross-cutting concerns and these cross-cutting concerns are conceptually
separate from the application's business logic. There are various common good examples
of aspects including logging, declarative transactions, security, caching, etc.
The key unit of modularity in OOP is the class, whereas in AOP the unit of modularity is
the aspect. DI helps you decouple your application objects from each other, while AOP
helps you decouple cross-cutting concerns from the objects that they affect.
The AOP module of Spring Framework provides an aspect-oriented programming
implementation allowing you to define method-interceptors and pointcuts to cleanly
decouple code that implements functionality that should be separated. We will discuss
more about Spring AOP concepts in a separate chapter
"""

# ------------------ RUN ------------------
result = chain.invoke({"text": text})

print("\n\n=========== FINAL OUTPUT ===========\n")
print(result)

# Optional: Print Graph Structure
chain.get_graph().print_ascii()