import torch
import torch.nn as nn

# Reference: The idea for Line 17 was taken from here: https://discuss.pytorch.org/t/how-to-write-custom-crossentropyloss/58072
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def soft_max(self, prediction):
        numerator = torch.exp(prediction)
        denominator = torch.sum(numerator, dim=1, keepdim=True)

        return numerator / denominator

    def nll_loss(self, probabilities, label):
            batch_size = probabilities.shape[0]
            return -probabilities[range(batch_size), label].log().mean()

    def forward(self, prediction, label):
        prob = self.soft_max(prediction)
        loss = self.nll_loss(prob, label)
        return loss
