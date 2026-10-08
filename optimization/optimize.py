#=========convert model to onnx=================================

from optimum.onnxruntime import ORTModelForSequenceClassification 
from transformers import AutoTokenizer

model_tokenizer_path="./models"
tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path)


onnx_model = ORTModelForSequenceClassification.from_pretrained(
    model_tokenizer_path,
    export=True,
    file_name="model.onnx"

)

onnx_model.save_pretrained(model_tokenizer_path) 




