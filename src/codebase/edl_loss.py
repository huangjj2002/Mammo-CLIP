"""
Evidential Deep Learning (EDL) 损失函数模块

实现基于Dirichlet分布的不确定性建模损失函数：
  - 类交叉熵损失 (Type II Maximum Likelihood / Expected Mean Square Error)
  - KL散度正则化
  - 组合损失

参考文献:
  Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018
"""

import torch
import torch.nn.functional as F
from torch import nn


def relu_evidence(y):
    """使用ReLU激活函数确保evidence非负"""
    return F.relu(y)


def exp_evidence(y):
    """使用指数激活函数确保evidence非负"""
    return torch.exp(torch.clamp(y, min=-10, max=10))


def softplus_evidence(y):
    """使用Softplus激活函数确保evidence非负（平滑版本的ReLU）"""
    return F.softplus(y)


def get_annealing_coef(epoch_num, annealing_start=0, annealing_epochs=10):
    """Linearly increase the KL annealing coefficient from 0.0 to 1.0."""
    if epoch_num < annealing_start:
        return 0.0
    epochs = max(float(annealing_epochs), 1.0)
    coef = (float(epoch_num) - float(annealing_start)) / epochs
    return min(1.0, coef)


def _reduce_loss(loss, reduction="mean"):
    if reduction == "none":
        return loss.squeeze(-1)
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def _compute_alpha(output):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    return evidence, alpha


def edl_mse_loss(
    output,
    target,
    num_classes,
    reduction="mean",
):
    """
    EDL 均方误差损失 (Expected Mean Square Error under Dirichlet)
    
    L_mse = Σ_j (y_j - α_j/S)^2 + α_j(S - α_j)/(S^2(S+1))
    正则化项使用KL散度，带退火系数
    
    Args:
        output: 模型输出的evidence [B, K]
        target: one-hot标签 [B, K]
        num_classes: 类别数
    """
    _, alpha = _compute_alpha(output)
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # 预测误差项
    err = torch.sum((target - alpha / S) ** 2, dim=1, keepdim=True)
    # 方差项
    var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    
    mse_loss = err + var
    return _reduce_loss(mse_loss, reduction=reduction)


def edl_digamma_loss(
    output,
    target,
    num_classes,
    reduction="mean",
):
    """
    EDL 类交叉熵损失 (Type II Maximum Likelihood with Digamma)
    
    L_ce = Σ_j y_j * (ψ(S) - ψ(α_j))
    正则化项使用KL散度，带退火系数
    
    Args:
        output: 模型输出的evidence [B, K]
        target: one-hot标签 [B, K]
        num_classes: 类别数
    """
    _, alpha = _compute_alpha(output)
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # 类交叉熵损失：使用 digamma 函数
    ce_loss = torch.sum(target * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    return _reduce_loss(ce_loss, reduction=reduction)


def edl_log_loss(
    output,
    target,
    num_classes,
    reduction="mean",
):
    """
    EDL 对数损失 (Expected Log Loss under Dirichlet)
    
    L_log = Σ_j y_j * (log(S) - log(α_j))
    """
    _, alpha = _compute_alpha(output)
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    log_loss = torch.sum(target * (torch.log(S + 1e-10) - torch.log(alpha + 1e-10)), dim=1, keepdim=True)
    return _reduce_loss(log_loss, reduction=reduction)


def kl_divergence(alpha, num_classes):
    """
    计算Dirichlet分布与均匀Dirichlet分布之间的KL散度
    
    KL(Dir(α) || Dir(1,...,1))
    
    Args:
        alpha: Dirichlet参数 [B, K]
        num_classes: 类别数K
    
    Returns:
        KL散度 [B, 1]
    """
    beta = torch.ones_like(alpha)  # 均匀Dirichlet参数全为1
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    
    return kl


class EDLLoss(nn.Module):
    """
    EDL损失函数模块
    
    支持三种损失类型：
      - 'digamma': 类交叉熵损失（默认，推荐）
      - 'log': 对数损失
      - 'mse': 均方误差损失
    """
    def __init__(
        self,
        num_classes=2,
        loss_type='digamma',
        kl_weight=0.1,
        annealing_start=0,
        annealing_epochs=10,
        annealing_step=None,
        class_weights=None,
    ):
        """
        Args:
            num_classes: 类别数
            loss_type: 损失类型 ('digamma', 'log', 'mse')
            kl_weight: KL正则项权重
            annealing_start: KL退火开始epoch
            annealing_epochs: KL退火到1.0所需epoch数
        """
        super(EDLLoss, self).__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.kl_weight = kl_weight
        self.annealing_start = annealing_start
        if annealing_step is not None and annealing_epochs == 10:
            annealing_epochs = annealing_step
        self.annealing_epochs = annealing_epochs
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if class_weights.numel() != num_classes:
                raise ValueError(
                    f"class_weights length {class_weights.numel()} does not match num_classes={num_classes}"
                )
        self.class_weights = class_weights
        
        if loss_type == 'digamma':
            self.loss_fn = edl_digamma_loss
        elif loss_type == 'log':
            self.loss_fn = edl_log_loss
        elif loss_type == 'mse':
            self.loss_fn = edl_mse_loss
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}, 请选择 'digamma', 'log' 或 'mse'")
    
    def forward(self, output, target):
        """
        Args:
            output: 模型输出的evidence [B, K]  (K=num_classes)
            target: 标签 [B]  (整数类别索引，0~K-1)
            
        Returns:
            loss: 标量损失值
        """
        # 将整数标签转换为one-hot编码
        output = output.float()

        if target.dim() == 1:
            target_onehot = F.one_hot(target.long(), num_classes=self.num_classes).float()
        else:
            target_onehot = target.float()
        target_onehot = target_onehot.to(device=output.device, dtype=output.dtype)
        
        epoch_num = getattr(self, 'current_epoch', 0)
        annealing_coef = get_annealing_coef(
            epoch_num,
            annealing_start=self.annealing_start,
            annealing_epochs=self.annealing_epochs,
        )

        per_sample_data_loss = self.loss_fn(
            output,
            target_onehot,
            self.num_classes,
            reduction="none",
        )

        _, alpha = _compute_alpha(output)
        kl_alpha = target_onehot + (1 - target_onehot) * alpha
        per_sample_kl = kl_divergence(kl_alpha, self.num_classes)
        kl_loss = per_sample_kl.mean()

        if self.class_weights is None:
            data_loss = per_sample_data_loss.mean()
        else:
            if target.dim() == 1:
                target_indices = target.long()
            else:
                target_indices = torch.argmax(target_onehot, dim=1)
            sample_weights = self.class_weights.to(output.device)[target_indices]
            weighted_loss = per_sample_data_loss * sample_weights
            data_loss = weighted_loss.sum() / sample_weights.sum().clamp_min(1e-8)

        return data_loss + self.kl_weight * annealing_coef * kl_loss
