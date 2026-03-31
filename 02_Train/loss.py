import torch
import torch.nn as nn


class CVALoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        return loss, loss, loss


if __name__ == "__main__":
    crit = CVALoss()
    pred = torch.tensor([[0.5]], dtype=torch.float32)
    target = torch.tensor([[0.4]], dtype=torch.float32)
    out = crit(pred, target)
    print(tuple(float(x) for x in out))
