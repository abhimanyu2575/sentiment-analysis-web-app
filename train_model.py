#import dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# loading datset
data = pd.read_csv(r"C:\Users\BIT\OneDrive\Desktop\side projects\movie.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42, shuffle=True)

# Build pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train model 
model.fit(X_train, y_train)

# Evaluate 
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
