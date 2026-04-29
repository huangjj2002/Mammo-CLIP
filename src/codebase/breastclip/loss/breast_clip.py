"""
Mammo-CLIP 核心对比损失函数
实现了多正样本对比学习损失，包含：
  - 4组图文对比（I1-T1, I2-T1, I1-T2, I2-T2）
  - 图像间对比损失 ICL（Image-Contrastive Loss）
  - 文本间对比损失 TCL（Text-Contrastive Loss）

变量来源说明：
  - image_embeddings:   来自 BreastClip 模型的 image_projection 输出，[B, proj_dim]
  - text_embeddings:    来自 BreastClip 模型的 text_projection 输出，[B, proj_dim]
  - text_embeddings2:   来自 BreastClip 模型对第二段文本的编码投影，[B, proj_dim]
  - image_view_embeddings: 来自同一患者另一视角图像的编码投影，[B, proj_dim]
  - labels:             batch 内样本的索引标签 [0, 1, 2, ..., B-1]
  - logit_scale:        可学习的温度参数，logit_scale = exp(log(1/temperature))
  - label_smoothing:    来自 YAML 配置 loss.breast_clip.label_smoothing，默认0.0
  - i2i_weight:         来自 YAML 配置 loss.breast_clip.i2i_weight，图像间对比损失权重
  - t2t_weight:         来自 YAML 配置 loss.breast_clip.t2t_weight，文本间对比损失权重
  - loss_ratio:         来自 YAML 配置 loss.breast_clip.loss_ratio，总损失缩放比例
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from breastclip import util

# 分布式训练中的全聚合函数（将所有 GPU 上的 tensor 收集到一起）
all_gather_func = util.DistAutogradAllGatherFunction(partial=False)


def all_gather(tensor):
    """
    在分布式训练中，将所有 GPU 上的 tensor 聚合。
    单卡训练时直接返回原 tensor。
    
    Args:
        tensor: 当前 GPU 上的张量，如 image_embeddings [B, proj_dim]
    Returns:
        all_tensor: 所有 GPU 拼接后的张量 [world_size*B, proj_dim]
    """
    world_size = util.GlobalEnv.get().world_size  # GPU 总数（单卡为1）
    if world_size > 1:
        tensor_list = all_gather_func.apply(tensor)
        all_tensor = torch.cat(tensor_list, 0)  # [world_size*B, proj_dim]
    else:
        all_tensor = tensor
    return all_tensor


class BreastClip(nn.Module):
    """
    Mammo-CLIP 的核心对比损失。
    
    设计思路：在乳腺X光场景中，一个患者有多个视角的图像（CC/MLO），
    一份报告有多个文本段落（findings/impression），因此存在多个正样本对。
    
    总损失 = 图文对比损失(I-T) + i2i_weight × 图像间对比(ICL) + t2t_weight × 文本间对比(TCL)
    """
    def __init__(self, label_smoothing=0.0, i2i_weight=0.0, t2t_weight=0.0, loss_ratio=1.0):
        """
        Args:
            label_smoothing (float): 标签平滑系数，防止模型过度自信。来自 YAML 配置。
            i2i_weight (float): 图像间对比损失(ICL)的权重。来自 YAML 配置，默认0表示不使用。
            t2t_weight (float): 文本间对比损失(TCL)的权重。来自 YAML 配置，默认0表示不使用。
            loss_ratio (float): 损失缩放比例，用于 CombinedLoss 中调节各损失的贡献。
        """
        super(BreastClip, self).__init__()
        self.name = "contrastive"
        self.label_smoothing = label_smoothing
        self.loss_ratio = loss_ratio
        self.i2i_weight = i2i_weight
        self.t2t_weight = t2t_weight

    def forward(self, image_embeddings, text_embeddings, text_embeddings2, image_view_embeddings, labels, logit_scale,
                is_train, **kwargs):
        """
        前向传播：计算多正样本对比损失。
        
        Args:
            image_embeddings:       图像1的嵌入向量 [B, proj_dim]，已 L2 归一化
            text_embeddings:        文本1的嵌入向量 [B, proj_dim]，已 L2 归一化
            text_embeddings2:       文本2的嵌入向量 [B, proj_dim]，已 L2 归一化
            image_view_embeddings:  图像2(另一视角)的嵌入向量 [B, proj_dim]，已 L2 归一化
            labels:                 样本标签 [B]，值为 0,1,...,B-1（对角线标签）
            logit_scale:            温度参数标量，logit_scale = exp(log(1/T))
            is_train:               是否在训练模式（影响 label_smoothing 和日志记录）
        Returns:
            loss: 总损失标量（需调用 .mean() 取均值）
        """
        world_rank = util.GlobalEnv.get().world_rank  # 当前 GPU 的 rank（单卡为0）
        batch_size = labels.size(0)  # 当前 GPU 上的 batch 大小

        all_image_embeddings = all_gather(image_embeddings)
        all_text_embeddings = all_gather(text_embeddings)
        all_text_embeddings2 = all_gather(text_embeddings2)
        all_image_view_embeddings = all_gather(image_view_embeddings)

        with torch.no_grad():
            labels = labels + (world_rank * batch_size)

        loss_i2t = 0
        loss_t2i = 0

        # I1 - T1
        logits_per_image = logit_scale * image_embeddings @ all_text_embeddings.T
        logits_per_text = logit_scale * text_embeddings @ all_image_embeddings.T

        label_smoothing = self.label_smoothing if is_train else 0.0
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        # I2 - T1
        logits_per_image = logit_scale * image_view_embeddings @ all_text_embeddings.T
        logits_per_text = logit_scale * text_embeddings @ all_image_view_embeddings.T

        label_smoothing = self.label_smoothing if is_train else 0.0
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        # I1 - T2
        logits_per_image = logit_scale * image_embeddings @ all_text_embeddings2.T
        logits_per_text = logit_scale * text_embeddings2 @ all_image_embeddings.T

        label_smoothing = self.label_smoothing if is_train else 0.0
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        # I2 - T2
        logits_per_image = logit_scale * image_view_embeddings @ all_text_embeddings2.T
        logits_per_text = logit_scale * text_embeddings2 @ all_image_view_embeddings.T

        label_smoothing = self.label_smoothing if is_train else 0.0
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        loss_i2t = loss_i2t / 4.0
        loss_t2i = loss_t2i / 4.0

        # ICL
        loss_i2i = 0

        logits_per_i2i1 = logit_scale * image_embeddings @ all_image_view_embeddings.T
        logits_per_i1i2 = logit_scale * image_view_embeddings @ all_image_embeddings.T

        loss_i2i += F.cross_entropy(logits_per_i2i1, labels)
        loss_i2i += F.cross_entropy(logits_per_i1i2, labels)

        loss_i2i = loss_i2i / 2.0

        # TCL
        loss_t2t = 0

        logits_per_t2t1 = logit_scale * text_embeddings2 @ all_text_embeddings.T
        logits_per_t1t2 = logit_scale * text_embeddings @ all_text_embeddings2.T

        loss_t2t += F.cross_entropy(logits_per_t2t1, labels)
        loss_t2t += F.cross_entropy(logits_per_t1t2, labels)

        loss_t2t = loss_t2t / 2.0

        if is_train:
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "loss/contrastive/steps_i2t", loss_i2t, util.GlobalEnv.get().summary_writer.global_step
            )
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "loss/contrastive/steps_t2i", loss_t2i, util.GlobalEnv.get().summary_writer.global_step
            )
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "loss/contrastive/steps_i2i", loss_i2i, util.GlobalEnv.get().summary_writer.global_step
            )
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "loss/contrastive/steps_t2t", loss_t2t, util.GlobalEnv.get().summary_writer.global_step
            )
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "params/logit_scale", logit_scale, util.GlobalEnv.get().summary_writer.global_step
            )
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "params/temperature", 1.0 / logit_scale, util.GlobalEnv.get().summary_writer.global_step
            )

        # contrastive loss
        loss = (loss_i2t + loss_t2i) / 2.0  # shape: (batch_size,)
        loss += loss_i2i * self.i2i_weight
        loss += loss_t2t * self.t2t_weight

        return loss.mean()
