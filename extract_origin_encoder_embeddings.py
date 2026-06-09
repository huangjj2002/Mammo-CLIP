"""
Export Mammo-CLIP origin image-encoder embeddings to fast local arrays.

Recommended output for repeated downstream reads:
  - embeddings.npy: row-aligned float array, load with np.load(..., mmap_mode="r")
  - metadata.csv: row-aligned image/patient metadata
  - manifest.json: reproducibility details

Use --origin-checkpoint when you want embeddings from a fine-tuned origin
classifier checkpoint. If omitted, the script exports the base Mammo-CLIP
encoder from --clip-chk-pt-path.
"""

import argparse
import importlib
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
CODEBASE_DIR = PROJECT_ROOT / "src" / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))
MODEL_MODULES_DIR = CODEBASE_DIR / "breastclip" / "model" / "modules"


DETECTOR_ARCHES = {
    "breast_clip_det_b5_period_n_ft",
    "breast_clip_det_b5_period_n_lp",
    "breast_clip_det_b2_period_n_ft",
    "breast_clip_det_b2_period_n_lp",
}
SWIN_NORM_ARCHES = {"swin_tiny_custom_norm", "swin_base_custom_norm"}


def import_model_module(module_name):
    package_name = "_mammoclip_model_modules"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(MODEL_MODULES_DIR)]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.{module_name}")


def load_custom_efficientnet(name):
    efficientnet_custom = import_model_module("efficientnet_custom")
    if name == "tf_efficientnetv2-detect":
        model = efficientnet_custom.EfficientNet.from_name("efficientnet-b2", num_classes=1)
        model.out_dim = 1408
        return model
    if name == "tf_efficientnet_b5_ns-detect":
        model = efficientnet_custom.EfficientNet.from_name("efficientnet-b5", num_classes=1)
        model.out_dim = 2048
        return model
    raise KeyError(f"Unsupported detector encoder: {name}")


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


class TimmEfficientNetMammo(nn.Module):
    def __init__(self, name, pretrained=False, in_chans=1):
        super().__init__()
        import timm

        model = timm.create_model(name, pretrained=pretrained, in_chans=in_chans)
        clsf = model.default_cfg["classifier"]
        n_features = model._modules[clsf].in_features
        model._modules[clsf] = nn.Identity()
        self.out_dim = n_features
        self.model = model
        self.pool = nn.Sequential(GeM(p=3, eps=1e-6), nn.Flatten())

    def forward(self, x):
        x = self.model.forward_features(x)
        return self.pool(x)


class TorchvisionResNet(nn.Module):
    def __init__(self, name):
        super().__init__()
        from torchvision.models.resnet import resnet101, resnet152

        if name == "resnet152":
            self.resnet = resnet152(pretrained=True)
        elif name == "resnet101":
            self.resnet = resnet101(pretrained=True)
        else:
            raise KeyError(f"Unsupported ResNet encoder: {name}")
        self.out_dim = 2048
        del self.resnet.fc

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        return torch.flatten(x, 1)


class HuggingfaceImageEncoder(nn.Module):
    def __init__(
        self,
        name="google/vit-base-patch16-224",
        pretrained=True,
        gradient_checkpointing=False,
        cache_dir="~/.cache/huggingface/hub",
        model_type="vit",
        local_files_only=False,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel, SwinModel, ViTModel

        self.model_type = model_type
        if pretrained:
            if self.model_type == "swin":
                self.image_encoder = SwinModel.from_pretrained(name)
            else:
                self.image_encoder = AutoModel.from_pretrained(
                    name,
                    add_pooling_layer=False,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
        else:
            model_config = AutoConfig.from_pretrained(name, cache_dir=cache_dir, local_files_only=local_files_only)
            if type(model_config).__name__ == "ViTConfig":
                self.image_encoder = ViTModel(model_config, add_pooling_layer=False)
            else:
                raise NotImplementedError(f"Not support training from scratch: {type(model_config).__name__}")

        if gradient_checkpointing and self.image_encoder.supports_gradient_checkpointing:
            self.image_encoder.gradient_checkpointing_enable()
        self.out_dim = self.image_encoder.config.hidden_size

    def forward(self, image):
        if self.model_type == "vit":
            output = self.image_encoder(pixel_values=image, interpolate_pos_encoding=True)
        elif self.model_type == "swin":
            output = self.image_encoder(pixel_values=image)
        else:
            raise KeyError(f"Unsupported HuggingFace image model type: {self.model_type}")
        return output["last_hidden_state"]


def load_image_encoder(config_image_encoder):
    source = config_image_encoder["source"].lower()
    name = config_image_encoder["name"].lower()

    if source == "huggingface":
        cache_dir = config_image_encoder.get("cache_dir", "~/.cache/huggingface/hub")
        local_cache = Path(cache_dir).expanduser() / f'models--{config_image_encoder["name"].replace("/", "--")}'
        return HuggingfaceImageEncoder(
            name=config_image_encoder["name"],
            pretrained=config_image_encoder["pretrained"],
            gradient_checkpointing=config_image_encoder.get("gradient_checkpointing", False),
            cache_dir=cache_dir,
            model_type=config_image_encoder.get("model_type", "vit"),
            local_files_only=local_cache.exists(),
        )

    if source == "cnn" and name in {"tf_efficientnetv2-detect", "tf_efficientnet_b5_ns-detect"}:
        return load_custom_efficientnet(name)
    if source == "cnn" and name in {"tf_efficientnet_b5_ns", "tf_efficientnetv2_s"}:
        return TimmEfficientNetMammo(name=config_image_encoder["name"])
    if source == "cnn" and name in {"resnet152", "resnet101"}:
        return TorchvisionResNet(name)

    raise KeyError(f"Not supported image encoder: {config_image_encoder}")


def normalize_image_array(img, img_path):
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        raise ValueError(f"Empty image array: {img_path}")

    img -= img.min()
    img_max = float(img.max())
    if not np.isfinite(img_max) or img_max <= 0.0:
        print(f"[WARN] zero dynamic range image encountered, replacing with zeros: {img_path}")
        img = np.zeros_like(img, dtype=np.float32)
    else:
        img /= img_max
    return np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)


def is_detector_arch(arch):
    return arch.lower() in DETECTOR_ARCHES


def build_fallback_tensor(args):
    width = int(args.img_size[0])
    height = int(args.img_size[1])
    if is_detector_arch(args.arch):
        img = np.zeros((height, width, 3), dtype=np.float32)
    else:
        img = np.zeros((height, width), dtype=np.float32)
    return torch.tensor((img - args.mean) / args.std, dtype=torch.float32)


class OriginEmbeddingDataset(Dataset):
    def __init__(self, args, df):
        self.args = args
        self.df = df.reset_index(drop=True)
        self.dir_path = args.data_dir / args.img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        patient_id = str(data["patient_id"])
        image_id = str(data["image_id"])

        img_path = self.dir_path / patient_id / image_id
        if not img_path.exists() and not image_id.endswith(".png"):
            img_path = self.dir_path / patient_id / f"{image_id}.png"

        try:
            width = int(self.args.img_size[0])
            height = int(self.args.img_size[1])
            if is_detector_arch(self.args.arch):
                img = Image.open(str(img_path)).convert("RGB")
                img = np.array(img)
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

            img = normalize_image_array(img, img_path)
            img = torch.tensor((img - self.args.mean) / self.args.std, dtype=torch.float32)
            is_fallback = False
            error = ""
        except Exception as exc:
            print(f"[WARN] fallback image used for {img_path}: {exc}")
            img = build_fallback_tensor(self.args)
            is_fallback = True
            error = str(exc)

        return {
            "x": img.unsqueeze(0),
            "y": torch.tensor(data[self.args.label], dtype=torch.long),
            "img_path": str(img_path),
            "is_fallback": is_fallback,
            "error": error,
        }


def collate_origin_embedding_batch(batch):
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "y": torch.from_numpy(np.array([item["y"] for item in batch], dtype=np.float32)),
        "img_path": [item["img_path"] for item in batch],
        "is_fallback": [bool(item.get("is_fallback", False)) for item in batch],
        "error": [item.get("error", "") for item in batch],
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Export origin Mammo-CLIP encoder embeddings.")
    parser.add_argument("--csv-path", required=True, help="Input CSV. Must contain patient_id and image_id.")
    parser.add_argument("--data-dir", required=True, help="Directory containing the image folder.")
    parser.add_argument("--img-dir", default="images_png", help="Image directory relative to --data-dir.")
    parser.add_argument(
        "--clip-chk-pt-path",
        default=str(PROJECT_ROOT / "model" / "b5-model-best-epoch-7.tar"),
        help="Mammo-CLIP checkpoint used to build the image encoder.",
    )
    parser.add_argument(
        "--origin-checkpoint",
        default=None,
        help="Optional fine-tuned origin classifier checkpoint, e.g. *_best_aucroc*.pth.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to embeddings/origin_encoder_<timestamp>.")
    parser.add_argument("--dataset", default="custom", help="Dataset name used by existing MammoDataset.")
    parser.add_argument("--arch", default="breast_clip_det_b5_period_n_ft", help="Origin classifier architecture.")
    parser.add_argument("--label", default="cancer", help="Label column. A dummy zero column is created if missing.")
    parser.add_argument("--img-size", nargs=2, type=int, default=[912, 1520], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--gpu-id", default=None, help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32", help="Storage dtype.")
    parser.add_argument("--l2-normalize", action="store_true", help="Save L2-normalized embeddings.")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "val", "test"],
        help="Optional split filter using the CSV split/fold columns.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Debug export limit.")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast during extraction.")
    parser.add_argument("--mean", type=float, default=0.3089279)
    parser.add_argument("--std", type=float, default=0.25053555408335154)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--p", type=float, default=1.0)
    return parser


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def strip_module_prefix(key):
    return key[len("module.") :] if key.startswith("module.") else key


def image_encoder_weights_from_state(state_dict):
    image_encoder_weights = {}
    for key, value in state_dict.items():
        key = strip_module_prefix(str(key))
        if key.startswith("image_encoder."):
            image_encoder_weights[key[len("image_encoder.") :]] = value
    return image_encoder_weights


def load_image_encoder_from_checkpoints(clip_checkpoint_path, origin_checkpoint_path=None):
    clip_checkpoint = load_torch_checkpoint(clip_checkpoint_path)
    image_encoder_config = clip_checkpoint["config"]["model"]["image_encoder"]
    image_encoder = load_image_encoder(image_encoder_config)

    base_weights = image_encoder_weights_from_state(checkpoint_state_dict(clip_checkpoint))
    if not base_weights:
        raise ValueError(f"No image_encoder.* weights found in {clip_checkpoint_path}")
    image_encoder.load_state_dict(base_weights, strict=True)
    loaded_source = str(clip_checkpoint_path)
    loaded_weight_count = len(base_weights)

    if origin_checkpoint_path is not None:
        origin_checkpoint = load_torch_checkpoint(origin_checkpoint_path)
        origin_weights = image_encoder_weights_from_state(checkpoint_state_dict(origin_checkpoint))
        if not origin_weights:
            raise ValueError(f"No image_encoder.* weights found in origin checkpoint: {origin_checkpoint_path}")
        image_encoder.load_state_dict(origin_weights, strict=True)
        loaded_source = str(origin_checkpoint_path)
        loaded_weight_count = len(origin_weights)

    return image_encoder, image_encoder_config, loaded_source, loaded_weight_count


def filter_dataframe(df, split, max_samples, label_col):
    df = df.copy().reset_index(drop=True)
    df["source_row"] = np.arange(len(df), dtype=np.int64)

    required = {"patient_id", "image_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    if label_col not in df.columns:
        print(f"[WARN] label column {label_col!r} is missing; creating a dummy zero label for dataset loading.")
        df[label_col] = 0

    if split != "all":
        split_values = df["split"].astype(str).str.strip().str.lower() if "split" in df.columns else pd.Series("", index=df.index)
        if split == "test":
            mask = (split_values == "test") | (df["fold"] == -1 if "fold" in df.columns else False)
        elif split == "val":
            mask = split_values == "val"
        else:
            if "fold" in df.columns:
                mask = (df["fold"] >= 0) & ~split_values.isin(["val", "test"])
            else:
                mask = ~split_values.isin(["val", "test"])
        df = df[mask].reset_index(drop=True)

    if max_samples is not None:
        df = df.head(max_samples).reset_index(drop=True)
    return df


def make_dataset_args(args, image_encoder_config):
    return SimpleNamespace(
        data_dir=resolve_path(args.data_dir),
        img_dir=args.img_dir,
        dataset=args.dataset,
        arch=args.arch,
        label=args.label,
        img_size=list(args.img_size),
        mean=float(args.mean),
        std=float(args.std),
        alpha=float(args.alpha),
        sigma=float(args.sigma),
        p=float(args.p),
        model_type="classifier",
        image_encoder_type=image_encoder_config.get("model_type", image_encoder_config.get("name")),
    )


def prepare_inputs(inputs, arch):
    arch = arch.lower()
    if arch in DETECTOR_ARCHES:
        return inputs.squeeze(1).permute(0, 3, 1, 2).contiguous()
    if arch in SWIN_NORM_ARCHES:
        return inputs.squeeze(1).contiguous()
    if inputs.dim() == 5:
        inputs = inputs.squeeze(1)
    if inputs.dim() == 4 and inputs.shape[-1] in (1, 3):
        inputs = inputs.permute(0, 3, 1, 2)
    return inputs.contiguous()


def encode_batch(image_encoder, image_encoder_config, inputs):
    encoder_type = image_encoder_config.get("model_type", "").lower()
    encoder_name = image_encoder_config.get("name", "").lower()

    if encoder_type == "cnn":
        if encoder_name in {"resnet152", "resnet101"}:
            features = image_encoder(inputs)
        elif encoder_name.endswith("-detect"):
            features = image_encoder({"image": inputs, "breast_clip_train_mode": True})
        else:
            features = image_encoder(inputs)
    else:
        features = image_encoder(inputs)

    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.dim() == 3:
        features = features[:, 0]
    if features.dim() != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {tuple(features.shape)}")
    return features


def metadata_columns(df, label_col):
    preferred = [
        "source_row",
        "patient_id",
        "image_id",
        "laterality",
        "view",
        "split",
        "fold",
        label_col,
        "cohort_num",
    ]
    return [col for col in preferred if col in df.columns]


def main():
    args = build_parser().parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    clip_checkpoint_path = resolve_path(args.clip_chk_pt_path)
    origin_checkpoint_path = resolve_path(args.origin_checkpoint) if args.origin_checkpoint else None
    csv_path = resolve_path(args.csv_path)
    output_dir = resolve_path(args.output_dir) if args.output_dir else PROJECT_ROOT / "embeddings" / (
        "origin_encoder_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[device] {device}")
    print(f"[csv] {csv_path}")
    print(f"[clip checkpoint] {clip_checkpoint_path}")
    if origin_checkpoint_path:
        print(f"[origin checkpoint] {origin_checkpoint_path}")
    else:
        print("[origin checkpoint] not provided; exporting base Mammo-CLIP encoder weights")

    image_encoder, image_encoder_config, loaded_source, loaded_weight_count = load_image_encoder_from_checkpoints(
        clip_checkpoint_path, origin_checkpoint_path
    )
    image_encoder = image_encoder.to(device).eval()
    feature_dim = int(getattr(image_encoder, "out_dim"))

    df = pd.read_csv(csv_path).fillna(0)
    df = filter_dataframe(df, args.split, args.max_samples, args.label)
    if len(df) == 0:
        raise ValueError("No rows left after filtering.")

    dataset_args = make_dataset_args(args, image_encoder_config)
    dataset = OriginEmbeddingDataset(args=dataset_args, df=df)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        collate_fn=collate_origin_embedding_batch,
    )

    storage_dtype = np.float16 if args.dtype == "float16" else np.float32
    embedding_path = output_dir / "embeddings.npy"
    embeddings = np.lib.format.open_memmap(
        embedding_path,
        mode="w+",
        dtype=storage_dtype,
        shape=(len(df), feature_dim),
    )

    meta = df[metadata_columns(df, args.label)].copy()
    meta.insert(0, "embedding_row", np.arange(len(meta), dtype=np.int64))
    meta["img_path"] = ""
    meta["is_fallback"] = False
    meta["error"] = ""

    offset = 0
    use_amp = bool(args.amp and device.type == "cuda")
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Exporting embeddings", total=len(loader)):
            inputs = prepare_inputs(batch["x"].to(device, non_blocking=args.pin_memory), args.arch)
            with torch.cuda.amp.autocast(enabled=use_amp):
                features = encode_batch(image_encoder, image_encoder_config, inputs)
                if args.l2_normalize:
                    features = F.normalize(features.float(), p=2, dim=1)

            features_np = features.float().cpu().numpy().astype(storage_dtype, copy=False)
            batch_size = features_np.shape[0]
            embeddings[offset : offset + batch_size] = features_np
            meta.loc[offset : offset + batch_size - 1, "img_path"] = list(batch["img_path"])
            meta.loc[offset : offset + batch_size - 1, "is_fallback"] = list(batch["is_fallback"])
            meta.loc[offset : offset + batch_size - 1, "error"] = list(batch["error"])
            offset += batch_size

    embeddings.flush()
    metadata_path = output_dir / "metadata.csv"
    meta.to_csv(metadata_path, index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "csv_path": str(csv_path),
        "data_dir": str(dataset_args.data_dir),
        "img_dir": args.img_dir,
        "clip_checkpoint": str(clip_checkpoint_path),
        "origin_checkpoint": str(origin_checkpoint_path) if origin_checkpoint_path else None,
        "loaded_encoder_weights_from": loaded_source,
        "loaded_encoder_weight_count": loaded_weight_count,
        "image_encoder_config": image_encoder_config,
        "arch": args.arch,
        "label": args.label,
        "split": args.split,
        "img_size": list(args.img_size),
        "mean": args.mean,
        "std": args.std,
        "embedding_file": str(embedding_path),
        "metadata_file": str(metadata_path),
        "num_rows": int(len(df)),
        "feature_dim": int(feature_dim),
        "dtype": args.dtype,
        "l2_normalize": bool(args.l2_normalize),
        "fallback_rows": int(meta["is_fallback"].sum()),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"  embeddings: {embedding_path} shape=({len(df)}, {feature_dim}) dtype={args.dtype}")
    print(f"  metadata:   {metadata_path}")
    print(f"  manifest:   {manifest_path}")
    print('  fast read:  embeddings = np.load("embeddings.npy", mmap_mode="r")')
    if manifest["fallback_rows"]:
        print(f"[WARN] fallback image rows: {manifest['fallback_rows']} (see metadata.csv)")


if __name__ == "__main__":
    main()
