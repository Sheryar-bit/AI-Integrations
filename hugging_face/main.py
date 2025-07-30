from transformers import pipeline

# model = pipeline(task="summarization", model="facebook/bart-large-cnn")
model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

response = model("Text to summarize: ")
text = "ChatGPT is a language model developed by OpenAI. It can generate human-like responses to a variety of inputs, including code and essays."
response = model(text)
print(response)
