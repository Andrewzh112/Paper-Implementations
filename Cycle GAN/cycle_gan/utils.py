import random
import numpy as np
import torch
import torchvision
import logging
from collections import deque


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class LambdaLR:
    def __init__(self, n_epochs, starting_epoch, decay_epoch):
        self.n_epochs = n_epochs
        self.decay_epoch = decay_epoch
        self.starting_epoch = starting_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.starting_epoch - self.decay_epoch)/(self.n_epochs - self.decay_epoch)


def save_images(image_tensor, file_name, size=(3, 224, 224)):
    image_tensor = (image_tensor + 1) / 2
    images = image_tensor.detach().cpu().view(-1, *size)
    torchvision.utils.save_image(images, file_name)


class ReplayBuffer:
    def __init__(self, buffer_size=50):
        self.buffer = deque([], maxlen=buffer_size)

    def sample(self, in_batch):
        batch_size = in_batch.size(0)

        buffer_idx = np.array(range(len(self.buffer)))
        random.shuffle(buffer_idx)

        batch_idx = np.array(range(len(in_batch)))
        random.shuffle(batch_idx)
        if len(self.buffer) > 0:
            num_sample = np.random.binomial(batch_size, 0.5)
        else:
            num_sample = 0
        num_batch = batch_size - num_sample

        if len(buffer_idx[:num_sample]) > 0:
          buffer_samples = [sample for i, sample in enumerate(self.buffer) if i in buffer_idx[:num_sample]]
        else:
          buffer_samples = []
        if len(batch_idx[:num_batch]) > 0:
          batch_samples = in_batch[batch_idx[:num_batch]]
        else:
          batch_samples = []

        return_batch = torch.stack([*buffer_samples, *batch_samples], dim=0)
        self.buffer.extend([*in_batch])

        return return_batch
