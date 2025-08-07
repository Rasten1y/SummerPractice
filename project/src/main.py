import sys
import torch
from torchvision import transforms
from PIL import Image
from train import get_model, BiasedRandomResizedCrop
from pathlib import Path
from glob import glob
import numpy as np


def load_models(checkpoints_dir='checkpoints', device='cpu'):
    models = []
    ckpt_paths = sorted(Path(checkpoints_dir).glob('mobilenetv3_fold*_f1*_loss*.pt'))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")
    for ckpt_path in ckpt_paths:
        model = get_model().to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def ensemble_predict(models, img_path, Kcrops=20, device='cpu'):
    img = Image.open(img_path).convert('RGB')
    tf = transforms.Compose([
        BiasedRandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
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
    return ('day' if p_day > 0.50 else 'night', p_day)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py <images_dir>")
        sys.exit(1)
    img_dir = sys.argv[1]
    models = load_models(device='cpu')

    for path in glob(f"{img_dir}/*.*"):
        label, p = ensemble_predict(models, path, Kcrops=20, device='cpu')
        print(f"{path:30s} → {label.upper():5s} (p_day={p:.2f})")