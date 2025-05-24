from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

from .image_processor import image_processor
from .llava_med import model as med_model

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


 # load llava_med
from llava.model.builder import load_pretrained_model
tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path='microsoft/llava-med-v1.5-mistral-7b',
        model_base=None,
        model_name='llava-med-v1.5-mistral-7b'
 )


@app.route('/response', methods=['POST']) #
def chatbot():
    data = request.form
    message = str(data['message'])
    chat_history = data['chat_history']
    print(message)
    print(chat_history)

    med_responces = []
    input_ids, attention_mask = med_model.create_conversations(message)

    if 'image' in request.files:
        file = request.files['image']
        filename = "original.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        image_processor.analyze_hne_image(filepath)     # patches are saved in temp_uploads folder

        for image_name in os.listdir(UPLOAD_FOLDER):
            image_path = os.path.join(UPLOAD_FOLDER, image_name)
            image_tensor = med_model.get_image_tensors(image_path)
            answer = med_model.generate_response(input_ids, attention_mask, image_tensor)
            med_responces.append({image_path : answer})
    else:
        answer = med_model.generate_response(input_ids, attention_mask)
        med_responces.append(answer)




    # history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

    # response = get_response(message, history_str)
    # print(response)

    return jsonify( { 'response' : med_responces } )

if __name__ == "__main__":
    app.run(debug=True, port=5000)