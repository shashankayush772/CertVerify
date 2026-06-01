"""
ResNet-18 + ELA forgery detection (CASIA v2.0-style checkpoint).

Windows users: pdf2image needs Poppler. Download from
https://github.com/oschwartz10612/poppler-windows/releases
and add the bin folder to your PATH.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


# --- Legacy small CNN (used by train.py) ------------------------------------

class CertForgeryNet(nn.Module):
    """Lightweight CNN for binary classification (authentic vs forged)."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def default_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# --- ResNet-18 checkpoint + inference ---------------------------------------

_RESNET_BUNDLE: Optional[Tuple[nn.Module, List[str], torch.device]] = None
_RESNET_LOAD_FAILED: bool = False


def _model_path() -> Path:
    return Path(__file__).resolve().parent / "model.pt"


def _load_resnet_bundle(
    device: torch.device,
) -> Optional[Tuple[nn.Module, List[str], torch.device]]:
    path = _model_path()
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain 'model_state_dict'")
    state = checkpoint["model_state_dict"]
    model = resnet18(weights=None)
    if "fc.weight" in state:
        n_out = state["fc.weight"].shape[0]
        n_in = state["fc.weight"].shape[1]
        model.fc = nn.Linear(n_in, n_out)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    classes = list(checkpoint.get("classes", ["authentic", "forged"]))
    print(f"[CertVerify] Model loaded successfully. Classes: {classes}")
    return model, classes, device


def pdf_to_image(pdf_path: str) -> str:
    """
    Convert the first page of a PDF to a temporary JPEG and return its path.
    Uses pdf2image first, falls back to Pillow.
    """
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(
            pdf_path, first_page=1, last_page=1, fmt="jpeg"
        )
        if not images:
            return ""
        fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        images[0].convert("RGB").save(out_path, "JPEG", quality=95)
        return out_path
    except Exception as e:
        print(f"[CertVerify] pdf2image failed: {e} — trying Pillow fallback")
        try:
            im = Image.open(pdf_path)
            if getattr(im, "n_frames", 1) > 1:
                im.seek(0)
            im = im.convert("RGB")
            fd, out_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            im.save(out_path, "JPEG", quality=95)
            return out_path
        except Exception as e2:
            print(f"[CertVerify] Pillow fallback also failed: {e2}")
            return ""


def detect_forgery(image_path: str) -> Dict[str, Any]:
    """
    Run ResNet-18 on ELA-preprocessed image.

    Returns {"label": "authentic" | "forged" | "unknown", "confidence": float}
    confidence is the probability of the predicted class (0.0 to 1.0).
    """
    global _RESNET_BUNDLE, _RESNET_LOAD_FAILED

    try:
        # --- ML Temporarily Disabled for Render Free Tier (OOM Prevention) ---
        print("[CertVerify] ML detection bypassed to prevent Render Free Tier OOM crash.")
        return {"label": "unknown", "confidence": 0.0}
        # ---------------------------------------------------------------------

        if _RESNET_LOAD_FAILED:
            return {"label": "unknown", "confidence": 0.0}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model once and cache it
        if _RESNET_BUNDLE is None:
            if not _model_path().is_file():
                print(
                    "[CertVerify] WARNING: model.pt not found — "
                    "ML forgery detection disabled."
                )
                _RESNET_LOAD_FAILED = True
                return {"label": "unknown", "confidence": 0.0}
            try:
                _RESNET_BUNDLE = _load_resnet_bundle(device)
            except Exception as exc:
                print(f"[CertVerify] WARNING: failed to load model.pt: {exc}")
                _RESNET_LOAD_FAILED = True
                return {"label": "unknown", "confidence": 0.0}

        model, classes, model_device = _RESNET_BUNDLE

        # Apply ELA preprocessing
        try:
            from backend.ml.ela import compute_ela
            ela_img = compute_ela(image_path)
        except Exception as e:
            print(f"[CertVerify] ELA failed, using raw image: {e}")
            ela_img = Image.open(image_path).convert("RGB").resize((224, 224))

        # Ensure RGB mode
        if ela_img.mode != "RGB":
            ela_img = ela_img.convert("RGB")

        # Apply transforms
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        batch = tfm(ela_img).unsqueeze(0).to(model_device)

        # Run inference
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        prob_authentic = float(probs[0].item())
        prob_forged = float(probs[1].item())

        # ── DEBUG — visible in your backend terminal ──────────────────────
        print(f"[ML DEBUG] classes      = {classes}")
        print(f"[ML DEBUG] prob[0] authentic = {prob_authentic:.4f}")
        print(f"[ML DEBUG] prob[1] forged    = {prob_forged:.4f}")
        # ─────────────────────────────────────────────────────────────────

        # Determine label by comparing probabilities directly
        # This avoids any argmax/class-order confusion
        if prob_forged > prob_authentic:
            label = "forged"
            confidence = prob_forged
        else:
            label = "authentic"
            confidence = prob_authentic

        print(f"[ML DEBUG] verdict = {label}  confidence = {confidence:.4f}")

        return {"label": label, "confidence": round(confidence, 4)}

    except Exception as exc:
        print(f"[CertVerify] ML inference error (non-fatal): {exc}")
        return {"label": "unknown", "confidence": 0.0}


def load_model(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> Tuple[CertForgeryNet, torch.device]:
    """Load legacy CertForgeryNet weights (used by train.py)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CertForgeryNet(num_classes=2)
    path = Path(checkpoint_path)
    if path.is_file():
        state = torch.load(path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        elif isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def predict_forgery(
    image_path: str | Path,
    checkpoint_path: str | Path,
    class_names: Tuple[str, ...] = ("authentic", "forged"),
) -> dict:
    """Run legacy CertForgeryNet on a raw RGB image (training pipeline)."""
    model, device = load_model(checkpoint_path)
    transform = default_image_transform()
    img = Image.open(image_path).convert("RGB")
    batch = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    pred_idx = int(torch.argmax(torch.tensor(probs)).item())
    label = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        },
    }
