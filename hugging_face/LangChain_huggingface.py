from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch
from transformers import pipeline
pipe = pipeline("text-generation",
                model="distilgpt2",
                device=0,
                truncation=True
                )

## Wrapping it for LangChain so LangChain treats it like an LLM
llm = HuggingFacePipeline(pipeline=pipe)

#This will be the Prompt Template
template = PromptTemplate.from_template("Explain {topic} in detail to a {age} year old to understand")

# chaining the template with the model so that prompt is formatted and then sent to the modelf
chain = template | llm
topic = input("Topic: ")
age = input("Age: ")

response = chain.invoke({"topic": topic, "age": age })
print(response)