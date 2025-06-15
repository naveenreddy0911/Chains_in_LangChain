from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative']=Field(description="Give the sentiment of the feedback text")
    
parser1=PydanticOutputParser(pydantic_object=Feedback)
parser2=StrOutputParser()

prompt1=PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative.\n{text}\n{format_instruction}",
    input_variables=["text"],
    partial_variables={'format_instruction':parser1.get_format_instructions()}
)

prompt2=PromptTemplate(
    template="Write an appropriate response to this positive feedback.\n{text}",
    input_variables=["text"]
)

prompt3=PromptTemplate(
    template="Write an appropriate response to this negative feedback.\n{text}",
    input_variables=["text"]
)

classifier_chain=prompt1|model|parser1

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2|model|parser2),
    (lambda x:x.sentiment=='negative',prompt3|model|parser2),
    RunnableLambda(lambda x:"Could not find sentiment.")
)

chain=classifier_chain|branch_chain

text="""
This is a terrible smartphone.
"""

result=chain.invoke({"text":text})
print(result)

chain.get_graph().print_ascii()