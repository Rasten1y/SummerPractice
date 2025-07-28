import sys
import torch
from torchvision import transforms
from PIL import Image
from train import get_model, BiasedRandomResizedCrop
from glob import glob
import numpy as np

def load_models(checkpoint_dir='checkpoints', k=5, device='cpu'):
    models = []
    for m in range(1, k+1):
        model = get_model().to(device)
        model.load_state_dict(torch.load(f"{checkpoint_dir}/mobilenetv3_fold{m}.pt", map_location=device))
        model.eval()
        models.append(model)
    return models

def ensemble_predict(models, img_path, Kcrops=20, device='cpu'):
    img = Image.open(img_path).convert('RGB')
    tf = transforms.Compose([
        BiasedRandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])
    probs = []
    for model in models:
        p_i = []
        for _ in range(Kcrops):
            crop = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(crop).squeeze(1)
                p_i.append(torch.sigmoid(logits).item())
        probs.append(np.mean(p_i))
    p_day = float(np.mean(probs))
    return ('day' if p_day > 0.5 else 'night', p_day)

if __name__ == '__main__':
    models = load_models(device='cpu')
    dir = sys.argv[1]
    for path in glob(f"{dir}/*.*"):
        label, p = ensemble_predict(models, path, Kcrops=20, device='cpu')
        print(f"{path:30s} → {label.upper():5s} (p_day={p:.2f})")