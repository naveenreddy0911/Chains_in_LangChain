from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Generate a detailed report on {topic}.",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Generate a 5 pointer summary from the text below:/n {text}",
    input_variables=["text"]
)

chain=prompt1|model|parser|prompt2|model|parser

result=chain.invoke({"topic":"Unemployment in India"})

print(result)

chain.get_graph().print_ascii()