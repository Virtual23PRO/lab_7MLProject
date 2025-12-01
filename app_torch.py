from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "microsoft/mpnet-base"  
device = torch.device("cpu")  

tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
base_model.eval().to(device)

model = torch.compile(base_model, mode="max-autotune")  

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    norm: float

app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    inputs = tokenizer(
        request.text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs).last_hidden_state
    norm = outputs.norm().item()
    return PredictResponse(norm=norm)
