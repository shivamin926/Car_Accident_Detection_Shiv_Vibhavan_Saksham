import torch
from PIL import Image
import io
from model.res_net import ResNetBinary

def process_image(body):
    #Loading model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetBinary()
    model.load_state_dict(torch.load("model/trained_weights/model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

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

    return pred