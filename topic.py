import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from fuzzywuzzy import fuzz

# Load the trained model and tokenizer
model_path = r"C:\Users\sowmi\hackathon\topic_model\model.safetensors"  # Path to your saved model directory
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# Load the label mapping from label_mapping.json
with open(r"C:\Users\sowmi\hackathon\topic_model\label_mapping.json", "r") as f:
    label_mapping = json.load(f)

# Reverse the label mapping to decode numerical predictions into topic names
id_to_label = {int(k): v for k, v in label_mapping.items()}

# Predefined keywords for each label
keyword_mapping = {
    "Refund Issue": ["refund", "return", "money back", "credit", "reimbursement"],
    "Technical Issue": ["error", "crash", "technical", "bug", "issue", "problem", "malfunction"],
    "Policy Complaint": ["policy", "terms", "conditions", "rules", "agreement"],
    "Call wait time": ["wait time", "on hold", "call delay", "long wait", "queue"],
    "Product Feedback": ["feedback", "review", "suggestion", "product opinion", "comment"],
    "General Complaint": ["complaint", "unhappy", "dissatisfied", "problem", "concern"],
    "Positive Feedback": ["excellent", "great", "amazing", "good service", "positive"],
    "Billing Issue": ["bill", "billing", "charge", "invoice", "payment problem"],
    "Subscription Issue": ["subscription", "renewal", "membership", "plan", "auto-renew"],
    "Delivery Issue": ["delivery", "shipment", "logistics", "delay", "tracking"],
    "Customer Service Praise": ["thank you", "good service", "helpful", "appreciate", "great support"],
}

def classify_text(input_text):
    """
    Classifies the input text into one of the predefined topics.

    Args:
        input_text (str): The text to classify.

    Returns:
        str: The name of the predicted topic.
    """
    # Check for predefined keywords with fuzzy matching
    input_lower = input_text.lower()
    for label, keywords in keyword_mapping.items():
        for keyword in keywords:
            # Use fuzzy matching with a threshold
            if fuzz.partial_ratio(keyword.lower(), input_lower) > 80:
                return label  # Return the label directly if a fuzzy match is found

    # If no keyword matches, use the BERT model for classification
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted topic index
    predicted_index = torch.argmax(logits, dim=1).item()

    # Decode the predicted index into the corresponding topic name
    topic_name = id_to_label[predicted_index]

    return topic_name

if __name__ == "__main__":
    # Prompt user for input text
    input_text = input("Enter the text to classify: ")

    # Classify the input text
    predicted_topic = classify_text(input_text)

    # Print the predicted topic
    print(f"The input text belongs to the topic: {predicted_topic}")
