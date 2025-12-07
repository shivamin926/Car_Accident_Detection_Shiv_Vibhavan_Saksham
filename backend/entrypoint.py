from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
import io
from model.detector import ResNetBinary

# --------------------------
# MODEL LOADING
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNetBinary()
#Load your trained weights
model.load_state_dict(torch.load("model/trained_weights/model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

app = FastAPI(title="Car Damage Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# HEALTH CHECK
# --------------------------
@app.get("/")
def root():
    return {"status": "online", "message": "FastAPI backend running"}

# --------------------------
# IMAGE PREDICTION ENDPOINT
# --------------------------
# @app.post("/predict-image")
# async def predict_image(file: UploadFile = File(...)):
#     img_bytes = await file.read()
#     img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

#     # Transform to tensor
#     import torchvision.transforms as transforms
#     transform = transforms.Compose([
#         transforms.Resize((300, 250)),
#         transforms.ToTensor(),
#     ])
#     x = transform(img).unsqueeze(0).to(DEVICE)

#     # Prediction
#     with torch.no_grad():
#         logits = model(x)
#         pred = torch.argmax(logits, dim=1).item()

#     # TEMP since model not included:
#     pred = 0  # 0 = good, 1 = damaged (replace when model plugged)

#     return {"prediction": pred}

@app.post("/predict-image")
async def predict_image(request: Request):
    body = await request.body()
    
    # Load image from bytes
    img = Image.open(io.BytesIO(body)).convert("RGB")

    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((300, 250)),
        transforms.ToTensor(),
    ])
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    return {"prediction": pred}

