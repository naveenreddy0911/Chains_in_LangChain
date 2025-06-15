from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

prompt1=PromptTemplate(
    template="Generate short and simple notes from the following text./n{text}",
    input_variables=["text"]
)

prompt2=PromptTemplate(
    template="Generate 5 question answers from the following text./n{text}",
    input_variables=["text"]
)

prompt3=PromptTemplate(
    template="Merge the provided notes and quiz into a single document./nNotes:/n{notes}/n/nQuiz:/n{quiz}",
    input_variables=["notes","quiz"]
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1|model|parser,
    'quiz':prompt2|model|parser
})

merge_chain=prompt3|model|parser

chain=parallel_chain|merge_chain

text="""
Batch learning involves training a machine learning model on a large, fixed dataset, while online learning updates the model incrementally with new data as it arrives. Batch learning is suitable for static datasets, while online learning is better for dynamic environments where the data is constantly changing. 
Batch Learning:
Data: Uses a large, pre-defined dataset. 
Model Training: Trains the model on the entire dataset at once. 
Updates: Does not update the model incrementally as new data arrives. 
Resource Usage: Can be more resource-intensive, especially with large datasets. 
Examples: Training a model on a large, static dataset like image classification. 
Online Learning:
Data: Learns from a stream of data that arrives sequentially. 
Model Training: Updates the model with each new data instance or mini-batch. 
Updates: Updates the model incrementally as new data arrives. 
Resource Usage: Can be more computationally efficient, especially with very large datasets. 
Examples: Real-time fraud detection, stock market prediction. 
Key Differences:
Data Handling:
Batch learning processes the entire dataset at once, while online learning processes data sequentially. 
Model Updating:
Batch learning updates the model periodically after processing batches of data, while online learning updates the model continuously with each new data instance. 
Resource Requirements:
Batch learning can be more resource-intensive for large datasets, while online learning can be more efficient. 
Flexibility:
Online learning is more flexible for handling changing or dynamic data, while batch learning is better for static datasets. 
Sensitivity to Data Quality:
Online learning is more sensitive to noise and poor data quality compared to batch learning. 
In essence, batch learning is like baking a cake from a pre-determined recipe, while online learning is like continuously tweaking a recipe based on new ingredients as they become available. 
"""

result=chain.invoke({"text":text})
print(result)

chain.get_graph().print_ascii()