from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I love using transformers!"))

#OutPut:
#[{'label': 'postive', 'score': 0.99..}]

#Not ideal as requires bhot zyada storage; will start with hugging face ssoon.