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


def _parse_bool(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"y", "yes", "true", "1"}:
            return True
        if normalized in {"n", "no", "false", "0"}:
            return False
        raise ValueError(f"{name} must be y/n or true/false.")
    return bool(value)


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
        focal_gamma=0.0,
        wrong_evidence_penalty_weight=0.0,
        wrong_evidence_margin=0.05,
        wrong_evidence_class_balanced=True,
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
        self.focal_gamma = float(focal_gamma or 0.0)
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative.")
        self.wrong_evidence_penalty_weight = float(wrong_evidence_penalty_weight or 0.0)
        if self.wrong_evidence_penalty_weight < 0:
            raise ValueError("wrong_evidence_penalty_weight must be non-negative.")
        self.wrong_evidence_margin = float(wrong_evidence_margin)
        if self.wrong_evidence_margin < 0:
            raise ValueError("wrong_evidence_margin must be non-negative.")
        self.wrong_evidence_class_balanced = _parse_bool(
            wrong_evidence_class_balanced,
            "wrong_evidence_class_balanced",
        )
        self.last_data_loss = None
        self.last_unweighted_data_loss = None
        self.last_class_weighted_data_loss = None
        self.last_focal_data_loss = None
        self.last_kl_loss = None
        self.last_wrong_evidence_penalty = None
        self.last_margin_violation_mean = None
        self.last_total_evidence_mean = None
        self.last_total_loss = None
        self.last_annealing_coef = None
        self.last_class_weights = None
        self.last_focal_factor_mean = None
        self.last_sample_weight_mean = None
        self.last_focal_weighted_denominator = None
        self.last_class_counts = None
        self.last_class_data_loss_means = None
        self.last_class_weighted_loss_means = None
        self.last_class_focal_weighted_loss_means = None
        self.last_class_focal_factor_means = None
        self.last_class_wrong_evidence_penalty_means = None
        self.last_weighted = class_weights is not None
        
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
            target_indices = target.long().to(output.device)
        else:
            target_onehot = target.float()
        target_onehot = target_onehot.to(device=output.device, dtype=output.dtype)
        if target.dim() != 1:
            target_indices = torch.argmax(target_onehot, dim=1).long().to(output.device)
        
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
        unweighted_data_loss = per_sample_data_loss.mean()

        evidence, alpha = _compute_alpha(output)
        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
        p_true = probs.gather(1, target_indices.view(-1, 1)).squeeze(1).clamp(1e-8, 1.0)
        true_class_mask = F.one_hot(target_indices, num_classes=self.num_classes).bool()
        wrong_probs = probs.masked_fill(true_class_mask, float("-inf"))
        p_wrong = wrong_probs.max(dim=1).values
        margin_violation = F.relu(p_wrong - p_true + self.wrong_evidence_margin)
        total_evidence = evidence.sum(dim=1)
        per_sample_wrong_evidence_penalty = margin_violation * total_evidence
        if self.wrong_evidence_class_balanced:
            wrong_penalty_class_means = []
            for class_idx in range(self.num_classes):
                class_mask = target_indices == class_idx
                if class_mask.any():
                    wrong_penalty_class_means.append(per_sample_wrong_evidence_penalty[class_mask].mean())
            if wrong_penalty_class_means:
                wrong_evidence_penalty = torch.stack(wrong_penalty_class_means).mean()
            else:
                wrong_evidence_penalty = per_sample_wrong_evidence_penalty.mean()
        else:
            wrong_evidence_penalty = per_sample_wrong_evidence_penalty.mean()

        if self.focal_gamma > 0:
            focal_factor = (1.0 - p_true).pow(self.focal_gamma)
        else:
            focal_factor = torch.ones_like(per_sample_data_loss)

        kl_alpha = target_onehot + (1 - target_onehot) * alpha
        per_sample_kl = kl_divergence(kl_alpha, self.num_classes)
        kl_loss = per_sample_kl.mean()

        if self.class_weights is None:
            sample_weights = torch.ones_like(per_sample_data_loss)
        else:
            sample_weights = self.class_weights.to(output.device)[target_indices]

        weighted_loss = per_sample_data_loss * sample_weights
        class_weighted_data_loss = weighted_loss.sum() / sample_weights.sum().clamp_min(1e-8)
        focal_data_loss = (per_sample_data_loss * focal_factor).mean()
        focal_weighted_loss = weighted_loss * focal_factor
        if self.class_weights is None:
            data_loss = focal_weighted_loss.mean()
            focal_weighted_denominator = torch.as_tensor(
                per_sample_data_loss.numel(),
                dtype=output.dtype,
                device=output.device,
            )
        else:
            focal_weighted_denominator = (sample_weights * focal_factor).sum().clamp_min(1e-8)
            data_loss = focal_weighted_loss.sum() / focal_weighted_denominator

        total_loss = data_loss + self.kl_weight * annealing_coef * kl_loss
        if self.wrong_evidence_penalty_weight > 0:
            total_loss = (
                total_loss
                + self.wrong_evidence_penalty_weight * annealing_coef * wrong_evidence_penalty
            )
        class_counts = []
        class_data_loss_means = []
        class_weighted_loss_means = []
        class_focal_weighted_loss_means = []
        class_focal_factor_means = []
        class_wrong_evidence_penalty_means = []
        for class_idx in range(self.num_classes):
            class_mask = target_indices == class_idx
            class_counts.append(int(class_mask.detach().sum().cpu()))
            if class_mask.any():
                class_data_loss_means.append(float(per_sample_data_loss[class_mask].detach().mean().cpu()))
                class_weighted_loss_means.append(float(weighted_loss[class_mask].detach().mean().cpu()))
                class_focal_weighted_loss_means.append(float(focal_weighted_loss[class_mask].detach().mean().cpu()))
                class_focal_factor_means.append(float(focal_factor[class_mask].detach().mean().cpu()))
                class_wrong_evidence_penalty_means.append(
                    float(per_sample_wrong_evidence_penalty[class_mask].detach().mean().cpu())
                )
            else:
                class_data_loss_means.append(float("nan"))
                class_weighted_loss_means.append(float("nan"))
                class_focal_weighted_loss_means.append(float("nan"))
                class_focal_factor_means.append(float("nan"))
                class_wrong_evidence_penalty_means.append(float("nan"))

        self.last_data_loss = float(data_loss.detach().cpu())
        self.last_unweighted_data_loss = float(unweighted_data_loss.detach().cpu())
        self.last_class_weighted_data_loss = float(class_weighted_data_loss.detach().cpu())
        self.last_focal_data_loss = float(focal_data_loss.detach().cpu())
        self.last_kl_loss = float(kl_loss.detach().cpu())
        self.last_wrong_evidence_penalty = float(wrong_evidence_penalty.detach().cpu())
        self.last_margin_violation_mean = float(margin_violation.detach().mean().cpu())
        self.last_total_evidence_mean = float(total_evidence.detach().mean().cpu())
        self.last_total_loss = float(total_loss.detach().cpu())
        self.last_annealing_coef = float(annealing_coef)
        self.last_focal_factor_mean = float(focal_factor.detach().mean().cpu())
        self.last_sample_weight_mean = float(sample_weights.detach().mean().cpu())
        self.last_focal_weighted_denominator = float(focal_weighted_denominator.detach().cpu())
        self.last_class_counts = class_counts
        self.last_class_data_loss_means = class_data_loss_means
        self.last_class_weighted_loss_means = class_weighted_loss_means
        self.last_class_focal_weighted_loss_means = class_focal_weighted_loss_means
        self.last_class_focal_factor_means = class_focal_factor_means
        self.last_class_wrong_evidence_penalty_means = class_wrong_evidence_penalty_means
        if self.class_weights is None:
            self.last_class_weights = None
            self.last_weighted = False
        else:
            self.last_class_weights = self.class_weights.detach().cpu().tolist()
            self.last_weighted = True
        return total_loss
