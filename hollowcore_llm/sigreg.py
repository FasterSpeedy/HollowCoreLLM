import torch
import torch.nn.functional as F


def sigreg_loss(embeddings: torch.Tensor, num_slices: int = 256, std_floor: float = 1.0) -> torch.Tensor:
    if embeddings.numel() == 0:
        return embeddings.new_zeros(())
    x = embeddings.float().reshape(-1, embeddings.size(-1))
    if x.size(0) < 2:
        return x.new_zeros(())
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(x.size(0) - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    decorrelation = off_diag.pow(2).mean()
    std = x.std(dim=0, unbiased=False)
    variance_floor = F.relu(std_floor - std).pow(2).mean()
    return decorrelation + variance_floor
