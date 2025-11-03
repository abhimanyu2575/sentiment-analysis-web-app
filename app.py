# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['message']
    prediction = model.predict([text])[0]
    label = 'positive' if prediction == 1 else 'negative'
    return render_template('index.html', prediction_text=f"Sentiment: {label}")

if __name__ == "__main__":
    app.run(debug=True)


