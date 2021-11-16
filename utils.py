import torch
import yaml
from easydict import EasyDict
import random

def load_yaml(path):
    """
    loads a YAML file
    :param path: (string) path to the configuration.yaml file to load
    :return: config file processed into a dictionary by EasyDict
    """
    file = yaml.load(open(path), Loader=yaml.FullLoader)
    config = EasyDict(file)

    return config

def add_random_noise(image, pixel_noise_probability=0.2):
    """
    Adds random noise to each pixel in an image
    :param image: MNIST image
    :param prob: prob to add noise to a pixel
    :return: noisy images
    """


    batch_size, c, height, width = image.shape
    noisy_image = torch.zeros_like(image)
    noisy_image = noisy_image.to("cuda")
    for bs in range(batch_size):
        for h in range(height):
            for w in range(width):
                if random.random() < pixel_noise_probability:
                    random_noise = torch.rand(1) * 1 + 0
                    random_noise = random_noise.to("cuda")
                    noisy_image[bs][0][h][w] = image[bs][0][h][w] + random_noise
                else:
                    noisy_image[bs][0][h][w] = image[bs][0][h][w]

    return image
    #return noisy_image