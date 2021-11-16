import torch



def add_random_noise(image, pixel_noise_probability=0.2):
    """
    :param image: MINST image
    :param pixel_noise_probability: probability of a pixel to be noisy
    :return: noisy image
    """

    