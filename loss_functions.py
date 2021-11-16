import torch
import torch.nn as nn

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def soft_max(self, prediction):
        numerator = torch.exp(prediction)
        denominator = torch.sum(numerator, dim=1, keepdim=True)

        return numerator / denominator

    def nll_loss(self, prediction):

        return torch.log(self.soft_max(prediction))

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        outputs = self.nll_loss(prediction)
        outputs = outputs[range(batch_size), label]
        loss = - torch.sum(outputs) / batch_size
        return loss