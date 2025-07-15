import onnxruntime as ort
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load ONNX model
session = ort.InferenceSession("models/onnx/crisis_detector.onnx")

def predict_crisis(text, return_confidence=False):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    logits = session.run(None, ort_inputs)[0]

    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    predicted_label = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs, axis=1)[0])

    if return_confidence:
        return predicted_label, confidence
    return predicted_label
