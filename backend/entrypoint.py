from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
from backend.process import process_image
from model.res_net import ResNetBinary

app = FastAPI(title="Car Damage Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "online", "message": "FastAPI backend running"}

@app.post("/predict-image")
async def predict_image(request: Request):
    body = await request.body()

    # Processing image and getting prediction
    pred = process_image(body)

    return {"prediction": pred}

