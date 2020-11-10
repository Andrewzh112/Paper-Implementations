from torchvision import transforms
import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import random


class CycleImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train', load_shape=256, target_shape=224):
        super().__init__()
        self.root = root
        self.mode = mode
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(load_shape),
                transforms.RandomCrop(target_shape),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.files_A = glob.glob(os.path.join(root, '%sA' % mode) + '/*.*')
        self.files_B = glob.glob(os.path.join(root, '%sB' % mode) + '/*.*')
        self.len_files_A, self.len_files_B = len(self.files_A), len(self.files_B)
        self.permsA, self.permsB = list(range(self.len_files_A)), list(range(self.len_files_B))
        if self.len_files_A != self.len_files_B:
            self.match_length()

    def match_length(self, renew=False):
        def shuffle_cut(indices, length):
            random.shuffle(indices)
            return indices[:length]
        if self.len_files_A > self.len_files_B:
            if renew:
                self.files_A = glob.glob(os.path.join(self.root, '%sA' % self.mode) + '/*.*')
                self.permsA = list(range(self.len_files_A))
            self.permsA = shuffle_cut(self.permsA, self.len_files_B)
        else:
            if renew:
                self.files_B = glob.glob(os.path.join(self.root, '%sB' % self.mode) + '/*.*')
                self.permsB = list(range(self.len_files_B))
            self.permsB = shuffle_cut(self.permsB, self.len_files_A)

    def __getitem__(self, index):
        image_A = self.transform(Image.open(self.files_A[self.permsA[index]]).convert('RGB'))
        image_B = self.transform(Image.open(self.files_B[self.permsB[index]]).convert('RGB'))
        if index == len(self) - 1 and self.len_files_A != self.len_files_B:
            self.match_length(renew=True)
        return image_A, image_B

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))
