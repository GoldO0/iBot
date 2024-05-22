from flask import Flask, render_template, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Initialize Firebase Admin SDK
cred = credentials.Certificate("acc.json")  # Replace with your service account key file
firebase_admin.initialize_app(cred, {'databaseURL': 'https://ibot-base-default-rtdb.firebaseio.com/'})
database = db.reference()

# Load tokenizer and model for GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Function to load conversation history from Firebase
def load_history():
    history_snapshot = database.child("conversation_history").get()
    if history_snapshot:
        return history_snapshot
    else:
        return []

# Function to save conversation history to Firebase
def save_history(history):
    database.child("conversation_history").set(history)

# Load conversation history
conversation_history = load_history()

# Function to generate bot responses
def generate_response(input_text, max_length=100):
    # Concatenate conversation history and current input
    context = " ".join([item['user_input'] + " " + item['bot_response'] for item in conversation_history])
    
    # Append current user input to the context
    context += " " + input_text
    
    # Tokenize the concatenated context
    input_ids = tokenizer.encode(context, return_tensors="pt", add_special_tokens=True)
    
    # Ensure input_ids length is within max_position_embeddings limit
    max_input_length = model.config.max_position_embeddings - max_length - 1
    if input_ids.size(1) > max_input_length:
        input_ids = input_ids[:, -max_input_length:]
    
    # Generate the bot response
    outputs = model.generate(
        input_ids, 
        max_length=input_ids.size(1) + max_length, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7, 
        top_k=50, 
        top_p=0.95, 
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    
    # Decode the generated response
    bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process the bot response to avoid repetitive sequences
    lines = bot_response.split('. ')
    filtered_lines = []
    for line in lines:
        if line not in filtered_lines:
            filtered_lines.append(line)
    bot_response = '. '.join(filtered_lines).strip()
    
    return bot_response

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Chat route
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']

    # Generate bot response
    bot_response = generate_response(user_input, max_length=280)

    # Update conversation history
    conversation_history.append({"user_input": user_input, "bot_response": bot_response})
    save_history(conversation_history)

    # Return JSON response
    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
