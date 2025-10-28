from flask import Flask, request, render_template
import pickle

# Load model and vectorizer
model_path = 'model.pkl'
vectorizer_path = 'vectorizer.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from form
    news_text = request.form['news']
    
    # Transform text using the vectorizer
    transformed_text = vectorizer.transform([news_text])
    
    # Predict using the model
    prediction = model.predict(transformed_text)
    
    output = "Fake News ❌" if prediction[0] == 0 else "Real News ✅"

    return render_template('index.html', prediction_text=f'Result: {output}')

if __name__ == "__main__":
    app.run(debug=True)
