"""3D CNN behavior classification (PyTorch).

Includes a small 3D ConvNet for clip classification and an inference wrapper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class Simple3DConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d((1,2,2))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d((2,2,2))
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BehaviorModel:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Simple3DConvNet(in_channels=3, num_classes=2)
        self.model.to(self.device)
        self.model.eval()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def predict(self, clip: np.ndarray) -> dict:
        """Predict behavior on a clip.

        clip: numpy array shape (T, H, W, C) with C=3
        Returns dict with 'label' and 'confidence'
        """
        # Normalize and convert to tensor shape (1, C, T, H, W)
        clip = clip.astype(np.float32) / 255.0
        clip = np.transpose(clip, (3, 0, 1, 2))
        tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            cls = int(probs.argmax())
            conf = float(probs[cls])
        return {"label": "shoplifting" if cls == 1 else "normal", "confidence": conf}
