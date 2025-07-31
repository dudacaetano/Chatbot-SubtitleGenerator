from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

'''def get_Chat_response(text, chat_history=[]):
    # Construa histórico no formato de diálogo
    prompt = """You are an expert assistant who only answers questions about Taylor Swift.  
                You know everything about her albums, songs, tours, personal life (public info), and career.  
                Always reply in a friendly and clear tone, as if you're talking to a fan.  
                If the question is not about Taylor Swift, politely say you can only talk about her.  
                Keep your answers concise, factual, and engaging.  
                Respond in English only, using simple and accurate language."""
    
    for msg in chat_history:
        prompt += f"Usuário: {msg['user']}\nBot: {msg['bot']}\n"
    
    prompt += f"Usuário: {text}\nBot:"
    
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response'''



def get_Chat_response(text):
    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
if __name__ == '__main__':
    app.run()