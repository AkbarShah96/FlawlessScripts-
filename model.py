import torch
import torch.nn as nn


class MNISTModel(nn.Module):
    def __init__(self, num_classes = 10, ):
        super(MNISTModel, self).__init__()

        sequence = [nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Flatten()]

        # kernel_size: odd kernal sizes are generally prefered and smaller generally better for computation efficiency
        # padding_size: Set to 1 so that the input and output dimensions are the same. Preverse pixels on the permiter of the image.
        # stride: Set to 1 to preverse more spatial information

        self.layers = nn.Sequential(*sequence)
        self.output_layer = nn.Linear(in_features=1568, out_features=10)    # in_features hardcoded atm, it can change if you change the parameters above, smarter way is to make layers individually and feed the shape based on output.

    def forward(self, image):
        features = self.layers(image)
        return self.output_layer(features)