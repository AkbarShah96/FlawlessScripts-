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
        # learning rate was selected based on my intuition of working with Adam 1e-3 to 1e-4 is generally a good range to start with!
        # if you model is underfitting or loss is not decreasing, you can increase the learning rate or if its overfitting you can try to decrease it!

        # Initialize Schedular
        self.schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode="min",
                                                                    factor=self.args.OPTIMIZATION.sche_decay_factor,
                                                                    patience=self.args.OPTIMIZATION.sche_patience,
                                                                    threshold=self.args.OPTIMIZATION.sche_threshold,
                                                                    verbose=True)
        self.custom_loss = CustomCrossEntropyLoss()

        self.soft_max = torch.nn.Softmax()
        self.torch_CE = torch.nn.CrossEntropyLoss()

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
                                                       batch_size=self.args.VAL.val_batch_size,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       pin_memory=True,
                                                       drop_last=False)

            test_loader = torch.utils.data.DataLoader(dataset=test_split,
                                                       batch_size=self.args.VAL.val_batch_size,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       pin_memory=True,
                                                       drop_last=False)


            return train_loader, val_loader, test_loader

    def train_model(self):
        """
        Main training loop!
        """
        best_accuracy = 0

        for epoch in range(self.args.OPTIMIZATION.num_epochs):
            running_accuracy = 0
            patience = 0

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

                with torch.no_grad():
                    torch_loss = self.torch_CE(prediction, gt)

                loss.backward()
                self.optimizer.step()
                print("Epoch:", epoch, "Iter:", iter, "Loss:", loss.item(), "Torch_loss:", torch_loss.item())

            if self.args.VAL.validation:
                # Use validation loss to save the best model! and stop if you do not improve for a certain number of epochs.
                self.validate_model()

            if running_accuracy > best_accuracy:
                best_accuracy = running_accuracy
                patience = 0
                # Save model weights here if you want
            else:
                patience += 1

            if patience >= 5:
                break

    def model_accuracy(self, correct, loader):
        return 100 * (correct / len(loader))

    def validate_model(self):
        with torch.no_grad():
            correct = 0
            self.model.eval()
            print("Validating Model")
            for iter, batch in enumerate(self.val_loader):

                # Transfer to GPU if available
                for key in range(len(batch)):
                    batch[key] = batch[key].to(self.device)

                image, gt = batch[0], batch[1]
                prediction = self.model(image)
                max_log_prob = prediction.argmax(dim=1, keepdim=True)
                correct += max_log_prob.eq(gt).sum().item()

            print("Validation Accuracy:", self.model_accuracy(correct, self.val_loader))

    def test_model(self):
        correct = 0
        self.model.eval()
        for iter, batch in enumerate(self.test_loader):

            # Transfer to GPU if available
            for key in range(len(batch)):
                batch[key] = batch[key].to(self.device)

            image, gt = batch[0], batch[1]
            prediction = self.model(image)
            max_log_prob = prediction.argmax(dim=1, keepdim=True)
            correct += max_log_prob.eq(gt).sum().item()

        print("Test Accuracy:", self.model_accuracy(correct, self.test_loader))

if __name__ == '__main__':
    config = load_yaml(os.path.join(os.getcwd(), "config.yaml"))
    config.DATA.data_path = os.path.join(os.getcwd(), "data")
    classify = Handwriting_Digit_Classification(config)
    classify.train_model()
    print("Trainig Complete")
    classify.test_model()
    print("Testing Complete")