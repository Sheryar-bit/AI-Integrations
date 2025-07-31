####### USING 3 DIFFERENT MODELS TO PRACTICE LANGCHAIN #################

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# to remove logs error
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

# model for summarization
summarize_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer = HuggingFacePipeline(pipeline=summarize_pipeline)

# model for refinement from the summarized content
refinement_pipeline = pipeline( "text2text-generation", model="google/flan-t5-small")
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

# Modesl fro question and answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

summary_temlate = PromptTemplate.from_template("Summarize the following text in a {length} way:\n\n{text}")

summarization_chain = summary_temlate | summarizer | refiner

text_to_summarize = input("\nEnter the text to Summarize \n")

length=input("\nEnter the length (short/medium/long): \n").strip().lower()
length_map = {"short":50, "medium":150, "long":300}
max_length = length_map.get(length.lower(),150)

summary = summarization_chain.invoke({"text": text_to_summarize, "length": length})

print("\n ### GENERATED SUMMARY ###")
print(summary)

while True:
    question = input("\nAsk a question about the summary (or type 'exit' to stop):\n").strip()

    if question.lower() == "exit":
        break
    elif not question:
        print("Please enter a valid question.")
        continue

    try:
        qa_result = qa_pipeline(question=question, context=summary)
        print("\n **Answer:**")
        print(qa_result["answer"])
    except Exception as e:
        print(f"\nError processing question: {e}")

# while True:
#     question = input("\nAsk a question about the summary (or type 'exit' to stop):\n")
#     if question.lower() == "exit":
#         break
#
#     qa_result = qa_pipeline(question=question, context=summary)
#
#     print("\n **Answer:**")
#     print(qa_result["answer"])