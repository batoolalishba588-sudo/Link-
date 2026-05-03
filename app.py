import pandas as pd
import pickle
from flask import Flask, request, render_template_string
from sklearn.ensemble import RandomForestClassifier

# 1. Ye hissa model file (model.pkl) banaye ga agar nahi hai
data = {'url_len': [10, 50, 100], 'dots': [1, 3, 5], 'label': [0, 1, 1]}
df = pd.DataFrame(data)
model = RandomForestClassifier()
model.fit(df[['url_len', 'dots']], df['label'])
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

app = Flask(__name__)

@app.route('/')
def home():
    return '<h2>🔍 Detector Ready!</h2><form action="/predict" method="post"><input name="url" placeholder="Paste URL here"><button>Check</button></form>'

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    features = [[len(url), url.count('.')]]
    prediction = model.predict(features)
    result = "⚠️ PHISHING!" if prediction[0] == 1 else "✅ SAFE"
    return f"<h3>Result: {result}</h3><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=True)