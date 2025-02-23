from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

import requests
import io

url = "https://drive.google.com/file/d/1lgEOfEV5CoM9b7-Rtx8ElhrvQI1tLLAd/view?usp=drive_link"
df = pd.read_csv(io.StringIO(requests.get(url).text))


# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Ingredients"])

def get_facts(ingredients_input):
    items = [item.strip() for item in ingredients_input.split(",")]
    n = len(items)
    input_vector = vectorizer.transform([ingredients_input])
    similarities = cosine_similarity(input_vector, X).flatten()
    top_indices = similarities.argsort()[-n:][::-1]
    return " ".join(df.iloc[top_indices]["Facts"])

@app.route("/get_facts", methods=["POST"])
def api_get_facts():
    data = request.get_json()
    ingredients_input = data.get("ingredients", "")
    if not ingredients_input:
        return jsonify({"error": "No ingredients provided"}), 400
    facts = get_facts(ingredients_input)
    return jsonify({"facts": facts})

if __name__ == "__main__":
    app.run(debug=True)
