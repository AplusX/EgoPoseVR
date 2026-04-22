# Reference: https://github.com/hiroyasuakada/UnrealEgo

import numpy as np
import torch
import torch.nn as nn


class MpjreLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MpjreLoss, self).__init__()
        self.reduction = reduction

    def bgdR(self, m1, m2):
        """Computes geodesic distance between batches of rotation matrices."""
        m = torch.bmm(m1, m2.transpose(1, 2))
        trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta) * (180.0 / torch.pi)
        return theta

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred, ytrue)
        if self.reduction == 'mean' or self.reduction == 'batchmean':
            return torch.mean(theta)
        elif self.reduction == 'sum':
            return torch.sum(theta)
        elif self.reduction == 'none':
            return theta
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")


class MpjpeLoss(nn.Module):
    def __init__(self):
        super(MpjpeLoss, self).__init__()

    def forward(self, pred_pose, gt_pose):
        distance = torch.linalg.norm(gt_pose - pred_pose, dim=-1, ord=2)
        return torch.mean(distance)


def procrustes_alignment(source, target):
    """
    Perform Procrustes alignment from source to target.

    Args:
        source: (B, J, 3)
        target: (B, J, 3)

    Returns:
        Aligned source points, shape (B, J, 3)
    """
    assert source.shape == target.shape

    source_centroid = torch.mean(source, dim=1, keepdim=True)
    target_centroid = torch.mean(target, dim=1, keepdim=True)
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    W = torch.bmm(target_centered.transpose(2, 1), source_centered)
    U, _, V = torch.svd(W)
    R = torch.bmm(U, V.transpose(2, 1))
    aligned_source = torch.bmm(source_centered, R)

    return aligned_source + target_centroid


def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) in batch using torch tensors.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = torch.sum(X1**2, dim=1).sum(dim=1)
    K = X1.bmm(X2.permute(0, 2, 1))
    U, s, V = torch.svd(K)

    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


def batch_compute_similarity_transform_numpy(S1, S2):
    device = S1.device
    b = S1.shape[0]

    s1_np = S1.detach().cpu().numpy()
    s2_np = S2.detach().cpu().numpy()

    results = []
    for i in range(b):
        s1_hat_i = compute_similarity_transform(s1_np[i], s2_np[i])
        s1_hat_i_torch = torch.from_numpy(s1_hat_i).to(device=device)[None, ...]
        results.append(s1_hat_i_torch)
    return torch.cat(results, dim=0)


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) using numpy arrays.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True

    assert(S2.shape[1] == S1.shape[1])

    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = np.sum(X1**2)
    K = X1.dot(X2.T)
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))

    scale = np.trace(R.dot(K)) / var1
    t = mu2 - scale * (R.dot(mu1))
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat
