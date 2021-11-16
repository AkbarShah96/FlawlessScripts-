import os
import sys
import torch
import torchvision
from utils import *
from model import *
from loss_functions import *
from torchvision import transforms

class Handwriting_Digit_Classification:
    def __init__(self, args):

        # Settings
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Dataset
        self.train_loader, self.val_loader, self.test_loader = self.split_training_dataset(self.args.DATA.data_path)

        # Initialize Model
        self.model = MNISTModel(num_classes=10)
        self.model.to(self.device)

        # Initialize Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.OPTIMIZATION.learning_rate)
        # Initialize Schedular
        self.schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode="min",
                                                                    factor=self.args.OPTIMIZATION.sche_decay_factor,
                                                                    patience=self.args.OPTIMIZATION.sche_patience,
                                                                    threshold=self.args.OPTIMIZATION.sche_threshold,
                                                                    verbose=True)
        self.custom_loss = CustomCrossEntropyLoss()

    def split_training_dataset(self, data_path):
            if not data_path:
                raise Exception("Please provide data_path.")

            # Only basic transforms functions for now, can add data augmentation here.
            transforms_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), 0.3081),
            ])


            training_data = torchvision.datasets.MNIST(root=data_path,
                                                       train=True,
                                                       transform=transforms_train,
                                                       download=True)

            test_split = torchvision.datasets.MNIST(root=data_path,
                                                       train=False,
                                                       transform=transforms_train,
                                                       download=True)


            train_split, val_split = torch.utils.data.random_split(dataset=training_data,
                                                                   lengths=[50000, 10000])

            train_loader = torch.utils.data.DataLoader(dataset=train_split,
                                                       batch_size=self.args.OPTIMIZATION.batch_size,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       pin_memory=True,
                                                       drop_last=False)

            val_loader = torch.utils.data.DataLoader(dataset=val_split,
                                                       batch_size=self.args.OPTIMIZATION.batch_size,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       pin_memory=True,
                                                       drop_last=False)

            test_loader = torch.utils.data.DataLoader(dataset=test_split,
                                                       batch_size=self.args.OPTIMIZATION.batch_size,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       pin_memory=True,
                                                       drop_last=False)


            return train_loader, val_loader, test_loader

    def soft_max(self, prediction):
        return

    def train_model(self):
        """
        Main training loop!
        """
        best_loss = float("inf")

        for epoch in range(self.args.OPTIMIZATION.num_epochs):
            running_loss = 0

            self.model.train()

            for iter, batch in enumerate(self.train_loader):

                # Transfer to GPU if available
                for key in range(len(batch)):
                    batch[key] = batch[key].to(self.device)

                image, gt = batch[0], batch[1]

                noisy_image = add_random_noise(image, pixel_noise_probability=0.2)

                self.optimizer.zero_grad()
                prediction = self.model(noisy_image)
                loss = self.custom_loss(prediction=prediction, label=gt)

                loss.backward()
                self.optimizer.step()
                print("Iter:", iter, "Loss:", loss.item())


if __name__ == '__main__':
    config = load_yaml(os.path.join(os.getcwd(), "config.yaml"))
    config.DATA.data_path = os.path.join(os.getcwd(), "data")
    classify = Handwriting_Digit_Classification(config)
    classify.train_model()