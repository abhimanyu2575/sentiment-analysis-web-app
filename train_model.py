# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# --- Sample dataset ---
# You can replace this with a real dataset (like IMDb reviews or Twitter sentiment)
# data = {
#     'text': [
#         'I love this product, it’s amazing!',
#         'This is the worst experience ever.',
#         'I am feeling great today!',
#         'I hate waiting in long lines.',
#         'The food was okay, not too bad.',
#         'Absolutely fantastic performance!',
#         'Terrible customer service.',
#         'I am so happy and excited!',
#         'This movie was boring.',
#         'That was an awesome surprise!'
#     ],
#     'label': ['positive', 'negative', 'positive', 'negative', 'neutral', 'positive', 'negative', 'positive', 'negative', 'positive']
# }

# df = pd.DataFrame(data)


# loading datset
data = pd.read_csv(r"C:\Users\BIT\OneDrive\Desktop\side projects\movie.csv")

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42, shuffle=True)

# --- Build pipeline ---
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# --- Train model ---
model.fit(X_train, y_train)

# --- Evaluate ---
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# --- Save model ---
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")
