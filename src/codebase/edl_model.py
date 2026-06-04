"""
Evidential Deep Learning (EDL) 分类器模型

基于预训练的 Mammo-CLIP 图像编码器构建的EDL分类器。
将原始线性分类头替换为EDL分类头，输出evidence值构造Dirichlet分布。

架构: 图像 → 预训练图像编码器(不冻结,全量微调) → 特征 → EDL分类头 → evidence [B, K]

Dirichlet参数: α_i = evidence_i + 1
概率: p_i = α_i / S,  S = Σα_i
不确定性: u = K / S
"""

import torch
import torch.nn.functional as F
from torch import nn

from breastclip.model.modules import load_image_encoder


class EDLClassifierHead(nn.Module):
    """
    EDL分类头
    
    将特征映射到K个evidence值，使用Softplus激活确保非负。
    可选添加Dropout和隐藏层提升表达能力。
    """
    def __init__(self, feature_dim, num_classes, dropout=0.0, hidden_dim=None):
        """
        Args:
            feature_dim: 输入特征维度（来自图像编码器的out_dim）
            num_classes: 类别数K（二分类=2）
            dropout: Dropout比率
            hidden_dim: 隐藏层维度，None则直接线性映射
        """
        super(EDLClassifierHead, self).__init__()
        
        if hidden_dim is not None:
            self.head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
                nn.Softplus()
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
                nn.Softplus()
            )
    
    def forward(self, x):
        """
        Args:
            x: 特征向量 [B, feature_dim]
        Returns:
            evidence: evidence值 [B, num_classes]，非负
        """
        return self.head(x)


class BreastClipEDLClassifier(nn.Module):
    """
    基于预训练Mammo-CLIP图像编码器的EDL分类器。
    
    流程: images [B,C,H,W] → image_encoder → features [B, out_dim] → EDLClassifierHead → evidence [B, K]
    
    与原始BreastClipClassifier的区别：
      - 分类头替换为EDL分类头（Softplus激活，输出evidence而非logits）
      - 输出K个类别的evidence（二分类K=2），而非1个logit
      - 可计算Dirichlet参数、概率分布和不确定性
    """
    def __init__(self, args, ckpt, num_classes=2, dropout=0.0, hidden_dim=None):
        """
        Args:
            args: 命令行参数对象（与原始BreastClipClassifier兼容）
            ckpt: 预训练权重字典
            num_classes: 类别数K（二分类=2，因为EDL需要为每个类别输出evidence）
            dropout: EDL分类头的Dropout比率
            hidden_dim: EDL分类头隐藏层维度
        """
        super(BreastClipEDLClassifier, self).__init__()
        
        # ===== 加载预训练图像编码器（与原始BreastClipClassifier完全一致）=====
        print(ckpt["config"]["model"]["image_encoder"])
        self.config = ckpt["config"]["model"]["image_encoder"]
        
        self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])
        
        # 从预训练权重中提取图像编码器权重
        image_encoder_weights = {}
        for k in ckpt["model"].keys():
            if k.startswith("image_encoder."):
                image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
        self.image_encoder.load_state_dict(image_encoder_weights, strict=True)
        
        self.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
        self.arch = args.arch.lower()
        
        # 不冻结backbone - 全量微调
        self.train_mode = getattr(args, "train_mode", "full").lower()
        freeze_backbone = str(getattr(args, "freeze_backbone", "n")).lower() == "y"
        self.freeze_image_encoder = freeze_backbone or self.train_mode in {"head_only", "freeze_backbone", "lp"}
        if self.freeze_image_encoder:
            print("EDL Model: image encoder frozen; training EDL head only")
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()
        else:
            print("EDL Model: image encoder and EDL head trainable (full fine-tuning)")
            for param in self.image_encoder.parameters():
                param.requires_grad = True
        
        # ===== EDL分类头（替换原始LinearClassifier）=====
        self.num_classes = num_classes
        self.edl_head = EDLClassifierHead(
            feature_dim=self.image_encoder.out_dim,
            num_classes=num_classes,
            dropout=dropout,
            hidden_dim=hidden_dim
        )
        
        self.raw_features = None
        self.pool_features = None
        
        print(f"EDL Classifier Head: feature_dim={self.image_encoder.out_dim}, "
              f"num_classes={num_classes}, dropout={dropout}, hidden_dim={hidden_dim}")

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_image_encoder", False):
            self.image_encoder.eval()
        return self
    
    def get_image_encoder_type(self):
        """返回图像编码器类型标识"""
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
        """
        用预训练的图像编码器提取图像特征
        
        Args:
            image: 输入图像 [B, C, H, W]
        Returns:
            image_features: 图像特征向量 [B, out_dim]
        """
        if self.image_encoder_type == "cnn":
            if self.config["name"].lower() == "resnet152" or self.config["name"].lower() == "resnet101":
                image_features = self.image_encoder(image)
                return image_features
            else:
                input_dict = {"image": image, "breast_clip_train_mode": True}
                image_features, raw_features = self.image_encoder(input_dict)
                self.raw_features = raw_features
                self.pool_features = image_features
                return image_features
        else:
            image_features = self.image_encoder(image)
            global_features = image_features[:, 0]
            return global_features
    
    def forward(self, images):
        """
        前向传播：图像 → 编码器特征 → EDL evidence
        
        Args:
            images: 输入图像 [B, C, H, W]
        Returns:
            evidence: evidence值 [B, K]，非负，用于构造Dirichlet分布
        """
        if self.image_encoder_type.lower() == "swin":
            images = self._prepare_swin_images(images)
        
        image_feature = self.encode_image(images)
        evidence = self.edl_head(image_feature)
        return evidence

    def load_bce_encoder_state(self, checkpoint, strict=True):
        state_dict = checkpoint.get("model", checkpoint)
        image_encoder_weights = {}
        for key, value in state_dict.items():
            if key.startswith("image_encoder."):
                image_encoder_weights[".".join(key.split(".")[1:])] = value

        if not image_encoder_weights:
            raise ValueError("BCE checkpoint does not contain any image_encoder.* weights.")

        self.image_encoder.load_state_dict(image_encoder_weights, strict=strict)
        return len(image_encoder_weights)

    def set_encoder_trainable(self, trainable):
        trainable = bool(trainable)
        self.freeze_image_encoder = not trainable
        for param in self.image_encoder.parameters():
            param.requires_grad = trainable
        if not trainable:
            self.image_encoder.eval()

    def set_edl_head_trainable(self, trainable):
        for param in self.edl_head.parameters():
            param.requires_grad = bool(trainable)
    
    @staticmethod
    def compute_dirichlet_params(evidence):
        """
        计算Dirichlet分布参数
        
        α_i = evidence_i + 1
        """
        return evidence + 1
    
    @staticmethod
    def compute_probabilities(evidence):
        """
        从evidence计算预测概率
        
        p_i = α_i / S = (evidence_i + 1) / Σ(evidence_j + 1)
        """
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        return alpha / S
    
    @staticmethod
    def compute_uncertainty(evidence):
        """
        计算总不确定性（认知不确定性）
        
        u = K / S = K / Σα_i
        """
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        K = evidence.shape[-1]
        return K / S
    
    @staticmethod
    def compute_class_evidence(evidence):
        """
        返回每个类别的evidence值（直接返回）
        """
        return evidence
