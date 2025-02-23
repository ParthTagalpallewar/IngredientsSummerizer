from flask import Flask, request, jsonify
import pandas as pd
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv("dataset.csv")

# Initialize BERT-based summarizer
summarizer = pipeline("summarization")

def get_facts(ingredients_input):
    items = [item.strip() for item in ingredients_input.split(",")]
    facts = " ".join(df[df["Ingredients"].isin(items)]["Facts"])
    if not facts:
        return "No relevant facts found."
    if(len(items) > 4):
        summary = summarizer(facts, max_length=300, min_length=150, do_sample=False)
    else:
        summary = summarizer(facts, max_length=200, min_length=50, do_sample=False)
    return summary[0]["summary_text"]

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
