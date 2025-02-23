from flask import Flask, request, jsonify
import torch
import pickle
from flask_cors import CORS
from model import WOOGPTModel,Block,MultiHeadAttention,Head,FeedForward

chars = ""
with open("starter/wizard.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
vocab_size = len(chars)

model = WOOGPTModel(vocab_size)
with open("starter/woo-model-v2.pkl", "rb") as f:
    model = pickle.load(f)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])


app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "http://127.0.0.1:5500"}})

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
    app.run(debug=True)
