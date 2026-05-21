import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.penalty_matrix = torch.tensor([
            [0.0, 1.0, 3.0],   # true: not depression
            [1.0, 0.0, 1.0],   # true: moderate
            [3.0, 1.0, 0.0]    # true: severe
        ])

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)

        penalty = self.penalty_matrix.to(logits.device)[labels]

        loss = torch.sum(probs * penalty, dim=1)

        return loss.mean()