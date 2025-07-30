from transformers import pipeline

classifier = pipeline('sentiment-analysis')

res = classifier("I am learning Hugging face and I'm sad! ")
print(res)

