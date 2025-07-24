# Problem Statement: Create an end to end NLP complete pipleline
# Given two user-submitted questions, determine if they are semantically the same and should be merged.

# 1-> data Acquistition(get data from any dataset (webscrap, API) )
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv("sample_quora_questions_fixed.csv")
df[['question1', 'question2', 'is_duplicate']].head()

# 2-> text preparing (Cleaning)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
df['question1'] = df['question1'].apply(clean_text)
df['question2'] = df['question2'].apply(clean_text)

# 3-> Text Preprocessing (Tokenization + Embedding)
model = SentenceTransformer('all-MiniLM-L6-v2')  #Hugging face Model
q1_embeddings = model.encode(df['question1'].tolist(), show_progress_bar=True)
q2_embeddings = model.encode(df['question2'].tolist(), show_progress_bar=True)

# #4-> Step 4: Feature Engineering (Generate features like cosine similarity between vectors.)
from sklearn.metrics.pairwise import cosine_similarity
def compute_similarity(q1_vec, q2_vec):
    return [cosine_similarity([a], [b])[0][0] for a, b in zip(q1_vec, q2_vec)]

df['similarity'] = compute_similarity(q1_embeddings, q2_embeddings)

# 5->  Step 5: Modeling (Train a classifier like Logistic Regression or XGBoost on similarity score)
X = df[['similarity']]
y = df['is_duplicate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

#6->Evaluation Intrinsic (metric: Accuracy, F1-Score)
print(classification_report(y_test, y_pred))

#7-> Step 7: Deployment Strategy (can deploy this with FastAPI or Flask)