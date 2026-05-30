"""
Prototype + Evidential Deep Learning classifier for Mammo-CLIP.

The Mammo-CLIP image encoder produces one image-level embedding z. The
prototype head converts distances to class-wise evidence and keeps per-prototype
details for CSV explanations.
"""

import torch
import torch.nn.functional as F
from torch import nn


class PrototypeEDLHead(nn.Module):
    """Class-wise prototype head that returns non-negative EDL evidence."""

    def __init__(
        self,
        feature_dim,
        num_classes=2,
        prototypes_per_class=4,
        temperature=1.0,
        normalize=True,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("PrototypeEDLHead currently supports binary classification only.")
        if prototypes_per_class <= 0:
            raise ValueError("prototypes_per_class must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

        prototypes = torch.empty(self.num_classes, self.prototypes_per_class, self.feature_dim)
        nn.init.normal_(prototypes, mean=0.0, std=0.02)
        self.prototypes = nn.Parameter(prototypes)
        self.prototype_evidence_logits = nn.Parameter(torch.zeros(self.num_classes, self.prototypes_per_class))

    def initialize_prototypes(self, prototypes):
        expected_shape = (self.num_classes, self.prototypes_per_class, self.feature_dim)
        if tuple(prototypes.shape) != expected_shape:
            raise ValueError(f"Expected prototype tensor shape {expected_shape}, got {tuple(prototypes.shape)}.")
        with torch.no_grad():
            self.prototypes.copy_(prototypes.to(device=self.prototypes.device, dtype=self.prototypes.dtype))

    def forward(self, features, return_details=False):
        if features.dim() != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected features [B, {self.feature_dim}], got {tuple(features.shape)}.")

        features_for_distance = features
        prototypes_for_distance = self.prototypes
        if self.normalize:
            features_for_distance = F.normalize(features_for_distance, p=2, dim=-1)
            prototypes_for_distance = F.normalize(prototypes_for_distance, p=2, dim=-1)

        flat_prototypes = prototypes_for_distance.reshape(-1, self.feature_dim)
        distances = torch.cdist(features_for_distance, flat_prototypes, p=2).pow(2)
        distances = distances.view(-1, self.num_classes, self.prototypes_per_class)

        similarity = torch.exp(-distances / self.temperature)
        prototype_weights = F.softplus(self.prototype_evidence_logits).unsqueeze(0)
        prototype_evidence = similarity * prototype_weights
        evidence = prototype_evidence.sum(dim=-1)

        if not return_details:
            return evidence

        alpha = evidence + 1
        strength = alpha.sum(dim=-1, keepdim=True)
        details = {
            "features": features,
            "prototype_distance": distances,
            "prototype_similarity": similarity,
            "prototype_evidence": prototype_evidence,
            "evidence": evidence,
            "alpha": alpha,
            "S": strength,
            "prob": alpha / strength,
            "uncertainty": self.num_classes / strength,
        }
        return evidence, details


class BreastClipPrototypeEDLClassifier(nn.Module):
    """Mammo-CLIP image encoder with a Prototype + EDL head."""

    def __init__(
        self,
        args,
        ckpt,
        num_classes=2,
        prototypes_per_class=4,
        temperature=1.0,
        normalize=True,
    ):
        super().__init__()
        from breastclip.model.modules import load_image_encoder

        print(ckpt["config"]["model"]["image_encoder"])
        self.config = ckpt["config"]["model"]["image_encoder"]
        self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])

        image_encoder_weights = {}
        for key in ckpt["model"].keys():
            if key.startswith("image_encoder."):
                image_encoder_weights[".".join(key.split(".")[1:])] = ckpt["model"][key]
        self.image_encoder.load_state_dict(image_encoder_weights, strict=True)

        self.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
        self.arch = args.arch.lower()

        self.train_mode = getattr(args, "train_mode", "full").lower()
        freeze_backbone = str(getattr(args, "freeze_backbone", "n")).lower() == "y"
        self.freeze_image_encoder = freeze_backbone or self.train_mode in {"head_only", "freeze_backbone", "lp"}
        if self.freeze_image_encoder:
            print("Prototype EDL Model: image encoder frozen; training prototype EDL head only")
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()
        else:
            print("Prototype EDL Model: image encoder and prototype EDL head trainable (full fine-tuning)")
            for param in self.image_encoder.parameters():
                param.requires_grad = True

        self.num_classes = num_classes
        self.prototype_head = PrototypeEDLHead(
            feature_dim=self.image_encoder.out_dim,
            num_classes=num_classes,
            prototypes_per_class=prototypes_per_class,
            temperature=temperature,
            normalize=normalize,
        )
        self.raw_features = None
        self.pool_features = None
        print(
            "Prototype EDL Head: "
            f"feature_dim={self.image_encoder.out_dim}, num_classes={num_classes}, "
            f"prototypes_per_class={prototypes_per_class}, temperature={temperature}, normalize={normalize}"
        )

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_image_encoder", False):
            self.image_encoder.eval()
        return self

    def get_image_encoder_type(self):
        return self.image_encoder_type

    def _prepare_swin_images(self, images):
        if images.dim() == 5:
            images = images.squeeze(1)
        if images.dim() == 4 and images.shape[1] in (1, 3):
            return images
        if images.dim() == 4 and images.shape[-1] in (1, 3):
            return images.permute(0, 3, 1, 2)
        return images

    def encode_image(self, image):
        if self.image_encoder_type == "cnn":
            if self.config["name"].lower() in {"resnet152", "resnet101"}:
                return self.image_encoder(image)
            input_dict = {"image": image, "breast_clip_train_mode": True}
            image_features, raw_features = self.image_encoder(input_dict)
            self.raw_features = raw_features
            self.pool_features = image_features
            return image_features

        image_features = self.image_encoder(image)
        return image_features[:, 0]

    def extract_features(self, images):
        if self.image_encoder_type.lower() == "swin":
            images = self._prepare_swin_images(images)
        return self.encode_image(images)

    def initialize_prototypes(self, prototypes):
        self.prototype_head.initialize_prototypes(prototypes)

    def forward(self, images, return_details=False):
        image_feature = self.extract_features(images)
        return self.prototype_head(image_feature, return_details=return_details)

    @staticmethod
    def compute_dirichlet_params(evidence):
        return evidence + 1

    @staticmethod
    def compute_probabilities(evidence):
        alpha = evidence + 1
        strength = torch.sum(alpha, dim=-1, keepdim=True)
        return alpha / strength

    @staticmethod
    def compute_uncertainty(evidence):
        alpha = evidence + 1
        strength = torch.sum(alpha, dim=-1, keepdim=True)
        return evidence.shape[-1] / strength
