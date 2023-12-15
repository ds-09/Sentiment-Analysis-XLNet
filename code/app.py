from flask import Flask, request, jsonify
import torch
from transformers import XLNetForSequenceClassification, XLNetTokenizer

app = Flask(__name__)

# Load your pre-trained XLNet model
model_path = "finetuned_model.pth"  # Update with your actual path
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=3)  # Assuming 3 labels
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load XLNet tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

# Add a simple route for the root path
@app.route('/')
def home():
    return 'XLNet Sentiment Analysis Flask App'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        text = data['text']

        # Tokenize and encode the input text
        inputs = tokenizer(text, return_tensors='pt', max_length=400, truncation=True)

        # Make a prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted class
        predicted_class = torch.argmax(logits, dim=1).item()

        # Assuming you have a list of class labels (e.g., ['Negative', 'Neutral', 'Positive'])
        class_labels = ['Negative', 'Neutral', 'Positive']
        predicted_label = class_labels[predicted_class]

        # Return the prediction in JSON format
        response = {'prediction': predicted_label}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
