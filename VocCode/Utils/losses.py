import torch
from Utils import ramps
import torch.nn.functional as F


class ConsistencyWeight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """

    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup


def CE_loss(input_logits, target_targets, ignore_index, temperature=1):
    if len(target_targets.shape) > 3:
        target_targets = torch.argmax(target_targets, dim=1)
    return F.cross_entropy(input_logits / temperature, target_targets, ignore_index=ignore_index)
    # return -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))


def semi_ce_loss(inputs, targets,
                 conf_mask=True, threshold=None,
                 threshold_neg=.0, temperature_value=1):
    # target => logit, input => logit
    pass_rate = {}
    if conf_mask:
        # for negative
        targets_prob = F.softmax(targets/temperature_value, dim=1)
        
        # for positive
        targets_real_prob = F.softmax(targets, dim=1)
        
        weight = targets_real_prob.max(1)[0]
        total_number = len(targets_prob.flatten(0))
        boundary = ["< 0.1", "0.1~0.2", "0.2~0.3",
                    "0.3~0.4", "0.4~0.5", "0.5~0.6",
                    "0.6~0.7", "0.7~0.8", "0.8~0.9",
                    "> 0.9"]

        rate = [torch.sum((torch.logical_and((i - 1) / 10 < targets_real_prob, targets_real_prob < i / 10)) == True)
                / total_number for i in range(1, 11)]

        max_rate = [torch.sum((torch.logical_and((i - 1) / 10 < weight, weight < i / 10)) == True)
                    / weight.numel() for i in range(1, 11)]

        pass_rate["entire_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, rate)]
        pass_rate["max_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, max_rate)]

        mask = (weight >= threshold)

        mask_neg = (targets_prob < threshold_neg)

        neg_label = torch.nn.functional.one_hot(torch.argmax(targets_prob, dim=1)).type(targets.dtype)
        if neg_label.shape[-1] != 21:
            neg_label = torch.cat((neg_label, torch.zeros([neg_label.shape[0], neg_label.shape[1],
                                                           neg_label.shape[2], 21 - neg_label.shape[-1]]).cuda()),
                                  dim=3)
        neg_label = neg_label.permute(0, 3, 1, 2)
        neg_label = 1 - neg_label
          
        if not torch.any(mask):
            neg_prediction_prob = torch.clamp(1-F.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))
            zero = torch.tensor(0., dtype=torch.float, device=negative_loss_mat.device)
            return zero, pass_rate, negative_loss_mat[mask_neg].mean()
        else:
            positive_loss_mat = F.cross_entropy(inputs, torch.argmax(targets, dim=1), reduction="none")
            positive_loss_mat = positive_loss_mat * weight

            neg_prediction_prob = torch.clamp(1-F.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))

            return positive_loss_mat[mask].mean(), pass_rate, negative_loss_mat[mask_neg].mean()
    else:
        raise NotImplementedError

def slerp(v1, v2, alpha, epsilon=1e-6):
    """
    Performs Spherical Linear Interpolation (Slerp) between two batches of vectors.
    
    Args:
        v1 (torch.Tensor): First batch of vectors, shape [B, C, H, W].
        v2 (torch.Tensor): Second batch of vectors, shape [B, C, H, W].
        alpha (float): Interpolation factor, between 0 and 1.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        torch.Tensor: The interpolated vectors.
    """
    # L2 normalize the vectors along the channel dimension
    v1_norm = F.normalize(v1, p=2, dim=1)
    v2_norm = F.normalize(v2, p=2, dim=1)

    # Calculate the dot product (cosine similarity)
    dot = (v1_norm * v2_norm).sum(dim=1, keepdim=True)
    
    # Clamp dot product to avoid numerical instability with acos
    dot = torch.clamp(dot, -1.0 + epsilon, 1.0 - epsilon)

    # Get the angle between the vectors
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Handle the case where vectors are nearly collinear
    # In this case, fall back to linear interpolation
    mask = sin_theta < epsilon
    
    # Slerp formula
    term1 = (torch.sin((1.0 - alpha) * theta) / sin_theta) * v1_norm
    term2 = (torch.sin(alpha * theta) / sin_theta) * v2_norm
    slerp_result = term1 + term2

    # Linear interpolation for collinear cases
    lerp_result = (1.0 - alpha) * v1_norm + alpha * v2_norm
    
    return torch.where(mask, lerp_result, slerp_result)

def feature_level_mseloss(pred_features, target_features):
    """
    Calculates MSE loss between two L2-normalized feature maps.
    """
    pred_features_norm = F.normalize(pred_features, p=2, dim=1)
    # Target is already normalized from Slerp
    return F.mse_loss(pred_features_norm, target_features)