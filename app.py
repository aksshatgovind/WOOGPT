from flask import Flask, request, jsonify
import torch
import pickle
import os
import requests
from flask_cors import CORS
from model import WOOGPTModel
from dotenv import load_dotenv  

# Load environment variables
load_dotenv()
MODEL_URL = os.getenv("MODEL_URL")

if not MODEL_URL:
    raise ValueError("MODEL_URL is not set in the .env file!")

os.makedirs("starter", exist_ok=True)

# Load model dynamically
model_path = "starter/woo-model-v2.pkl"
if not os.path.exists(model_path):
    print("Downloading model from:", MODEL_URL)
    response = requests.get(MODEL_URL)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Model downloaded!")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load character mappings
with open("starter/wizard.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
vocab_size = len(chars)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(model.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    
    return jsonify({"response": generated_chars})

if __name__ == "__main__":
    app.run()
