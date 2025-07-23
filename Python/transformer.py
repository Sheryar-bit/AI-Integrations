from transformers import pipeline

classifier = pipeline("sentiment-analysis")   # Load the sentiment analysis model
user_input = input("Enter a sentence to analyze sentiment ")
result = classifier(user_input)# Analyze the sentiment

print(f"Sentiment: {result[0]['label']}, Confidence: {result[0]['score']:.2f}")

##result:
# Enter a sentence to analyze sentiment
# hello i am sick today feel too low
# Sentiment: {NEGATIVE, Confidence: 1.00}

#Not ideal as requires bhot zyada storage; will start with hugging face ssoon.