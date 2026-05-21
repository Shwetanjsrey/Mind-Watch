import torch
import torch.nn as nn


def get_class_weights():
    counts = torch.tensor([0.40, 0.51, 0.09])
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return weights


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=get_class_weights())

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)