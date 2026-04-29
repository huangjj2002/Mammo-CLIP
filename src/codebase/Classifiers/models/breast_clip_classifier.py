"""
Mammo-CLIP 下游分类器模型
基于预训练的 Mammo-CLIP 图像编码器构建的线性分类器。

架构: 图像 → 预训练图像编码器（可冻结） → 线性分类头 → logits

两种模式:
  - Fine-tune（_ft 后缀）: 图像编码器参数可训练，端到端更新
  - Linear Probe（_lp 后缀）: 图像编码器冻结，只训练分类头

变量来源:
  - args: 命令行参数，来自 train_classifier.py
  - args.arch: 架构名（决定是否冻结backbone），如 "breast_clip_det_b5_period_n_ft"
  - ckpt: 预训练权重字典，来自 torch.load(args.clip_chk_pt_path)
    - ckpt["config"]["model"]["image_encoder"]: 图像编码器配置（名称、类型等）
    - ckpt["model"]: 模型 state_dict（键名以 "image_encoder." 开头的为编码器权重）
  - n_class: 输出类别数（1=二分类, 4=density, 3=birads）
    - 来自 train_loop() 中根据 args.label 自动推断
"""

from torch import nn

from breastclip.model.modules import load_image_encoder, LinearClassifier


class BreastClipClassifier(nn.Module):
    """
    基于预训练图像编码器的下游分类器。
    
    流程: images [B,C,H,W] → image_encoder → features [B, out_dim] → LinearClassifier → logits [B, n_class]
    """
    def __init__(self, args, ckpt, n_class):
        """
        Args:
            args: 命令行参数对象
                  - args.arch: 模型架构名，决定是否冻结backbone
            ckpt: 预训练权重字典，包含:
                  - ckpt["config"]["model"]["image_encoder"]: 编码器配置
                  - ckpt["model"]: 完整模型 state_dict
            n_class (int): 输出类别数
                  - 1: 二分类（如 cancer），使用 BCEWithLogitsLoss
                  - 4: density 4分类，使用 CrossEntropyLoss
                  - 3: birads 3分类，使用 CrossEntropyLoss
        """
        super(BreastClipClassifier, self).__init__()

        # 从预训练权重的配置中读取图像编码器配置
        print(ckpt["config"]["model"]["image_encoder"])
        self.config = ckpt["config"]["model"]["image_encoder"]
        
        # 根据配置创建图像编码器（可能是 EfficientNet-B5/B2, ResNet, Swin等）
        self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])
        
        # 从预训练权重中提取图像编码器部分的权重（键名去掉 "image_encoder." 前缀）
        image_encoder_weights = {}
        for k in ckpt["model"].keys():
            if k.startswith("image_encoder."):
                image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
        self.image_encoder.load_state_dict(image_encoder_weights, strict=True)
        
        # 记录编码器类型（"cnn" 或 "swin"），影响 encode_image 的处理逻辑
        self.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
        self.arch = args.arch.lower()
        
        # Linear Probe 模式：冻结图像编码器，只训练分类头
        if (
                args.arch.lower() == "breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "breast_clip_det_b2_period_n_lp"):
            print("freezing image encoder to not be trained")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # 线性分类头：feature_dim → n_class
        # feature_dim 取决于编码器：EfficientNet-B5=2048, B2=1408
        self.classifier = LinearClassifier(feature_dim=self.image_encoder.out_dim, num_class=n_class)
        self.raw_features = None   # EfficientNet 的中间特征（用于检测任务）
        self.pool_features = None  # EfficientNet 的池化后特征

    def get_image_encoder_type(self):
        """返回图像编码器类型标识（"cnn" 或 "swin"）"""
        return self.image_encoder_type

    def encode_image(self, image):
        """
        用预训练的图像编码器提取图像特征。
        
        Args:
            image: 输入图像张量 [B, C, H, W]
            
        Returns:
            image_features: 图像特征向量 [B, out_dim]
                - CNN编码器: 全局平均池化后的特征 [B, 2048]
                - Swin编码器: [CLS] token 的特征 [B, hidden_dim]
        """
        if self.image_encoder_type == "cnn":
            if self.config["name"].lower() == "resnet152" or self.config["name"].lower() == "resnet101":
                # ResNet: 直接前向传播得到特征
                image_features = self.image_encoder(image)
                return image_features
            else:
                # EfficientNet: 使用 breast_clip_train_mode 返回中间特征
                input_dict = {"image": image, "breast_clip_train_mode": True}
                image_features, raw_features = self.image_encoder(input_dict)
                self.raw_features = raw_features    # 保存中间特征（可用于可视化/检测）
                self.pool_features = image_features  # 保存池化特征
                return image_features
        else:
            # Swin/ViT 编码器: 取 [CLS] token 作为全局特征
            image_features = self.image_encoder(image)
            global_features = image_features[:, 0]  # [CLS] token，shape: [B, hidden_dim]
            return global_features

    def forward(self, images):
        """
        前向传播：图像 → 编码器特征 → 分类 logits。
        
        Args:
            images: 输入图像 [B, C, H, W]
                    注意：调用前可能已经过 squeeze+permute 预处理
                    
        Returns:
            logits: 分类 logits [B, n_class]
                    - 二分类时为 [B, 1]，需要 sigmoid 后得到概率
                    - 多分类时为 [B, n_class]，需要 softmax 后得到概率
        """
        if self.image_encoder_type.lower() == "swin":
            # Swin 需要 [B, H, W, C] 格式，做维度转换
            images = images.squeeze(1).permute(0, 3, 1, 2)
        # 提取图像特征 → 分类
        image_feature = self.encode_image(images)
        logits = self.classifier(image_feature)
        return logits
