from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

model_name = "microsoft/mpnet-base" 
tokenizer = AutoTokenizer.from_pretrained(model_name)

ort_session = ort.InferenceSession(
    "model_optimized.onnx",
    providers=["CPUExecutionProvider"],
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    norm: float

app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    tokens = tokenizer(
        request.text,
        padding=True,
        truncation=True,
        return_tensors="np", 
    )

    inputs_onnx = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }

    
    outputs = ort_session.run(None, inputs_onnx)  
    last_hidden = outputs[0]                     
    norm = float(np.linalg.norm(last_hidden))

    return PredictResponse(norm=norm)