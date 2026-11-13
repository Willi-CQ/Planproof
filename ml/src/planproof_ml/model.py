from __future__ import annotations
import torch
from torch import nn
import segmentation_models_pytorch as smp
from .schema import MASK_CLASSES, INSTANCE_CLASSES, ROOM_TYPES, schema_digest

OUTPUTS = ("masks", "boundary", "centers", "boxes", "types")


class Detector(nn.Module):
    def __init__(self, encoder_weights="imagenet"):
        super().__init__()
        base = smp.Unet(encoder_name="resnet34", encoder_weights=encoder_weights, in_channels=3, classes=5)
        self.encoder = base.encoder
        self.decoder = base.decoder
        channels = 16
        self.mask_head = nn.Conv2d(channels, len(MASK_CLASSES), 1)
        self.boundary_head = nn.Conv2d(channels, 1, 1)
        self.center_head = nn.Conv2d(channels, len(INSTANCE_CLASSES), 1)
        self.box_head = nn.Conv2d(channels, 12, 1)
        self.type_head = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, len(ROOM_TYPES), 1)
        )
        nn.init.constant_(self.center_head.bias, -4.595)  # sparse full-resolution center prior ~1%
        with torch.no_grad():
            self.box_head.bias.copy_(torch.tensor([-2.944, -2.944, 0.0, 0.0] * 3))

    def forward(self, x):
        features = self.encoder(x)
        decoded = self.decoder(features)
        return {
            "masks": self.mask_head(decoded),
            "boundary": self.boundary_head(decoded),
            "centers": self.center_head(decoded),
            "boxes": self.box_head(decoded),
            "types": self.type_head(decoded),
        }

    def metadata(self):
        return {
            "architecture": "unet-resnet34-multihead",
            "schema_digest": schema_digest(),
            "outputs": OUTPUTS,
            "mask_classes": MASK_CLASSES,
            "instance_classes": INSTANCE_CLASSES,
            "room_types": ROOM_TYPES,
            "image_normalization": "imagenet_rgb",
            "box_encoding": "per-class normalized width,height + tanh half-pixel offsets",
        }


class ExportModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        result = self.model(image)
        return tuple(result[k] for k in OUTPUTS)


def load_detector(checkpoint, device="cpu"):
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    if data["model_metadata"]["schema_digest"] != schema_digest():
        raise ValueError("Checkpoint label/schema mismatch")
    model = Detector(encoder_weights=None).to(device)
    model.load_state_dict(data["model"], strict=True)
    model.eval()
    return model, data
