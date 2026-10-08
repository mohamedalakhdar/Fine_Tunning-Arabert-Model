import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device= "cuda"

tokenizer= AutoTokenizer.from_pretrained("./weights")
arabert_model= AutoModelForSequenceClassification.from_pretrained("./weights").to(device)

text="انه ممتاز"

inputs = tokenizer([text], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = arabert_model(**inputs)

logits= outputs.logits

model_prediction= torch.argmax(logits, dim=1).item()

def predict(predict):
    if predict == 0:
        return"negative"
    elif predict == 1:
        return "positive"
    elif predict == 2:
        return "mixed"

print(predict(predict=model_prediction))
