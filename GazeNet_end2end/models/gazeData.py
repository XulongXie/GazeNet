import os
import os.path

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class GazeMe(data.Dataset):
    # __init__ method will read data file
    def __init__(self, dir_path, part, transform=None):
        # initialize the properties inherited from the parent class
        super(GazeMe, self).__init__()
        # absolute path of all pictures
        assert part in ["train", "test", "val"]
        sample = []
        with open(os.path.join(dir_path, part, "label.txt")) as f:
            for line in f:
                # remove useless spaces at the beginning and end
                line = line.rstrip()
                # separate pictures and tags according to spaces
                words = line.split()
                sample.append((words[0], int(words[1])))
                self.sample = sample
                self.transform = transform

    # __len__ method returns the size of the custom data set, which is convenient for later traversal
    def __len__(self):
        # return the length of datasets
        return len(self.sample)

    # __getitem__ method supports subscript access
    def __getitem__(self, index):
        img, label = self.sample[index]
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_train_set(data_dir):
    trans_train = transforms.Compose([
        transforms.Grayscale(1),
        # this also convert pixel value from [0,255] to [0,1]
        transforms.ToTensor(),
        # 这个还要修改
        transforms.Normalize(mean = 0.4862,
                             std = 0.1690),
    ])
    # load dataset
    # 'C:/Users/hasee/Desktop/Master Project/Step1/Label'
    train_set = GazeMe(dir_path=data_dir, part="train", transform=trans_train)
    print("train set successfully read!")
    return train_set


def get_val_set(data_dir):
    trans_train = transforms.Compose([
        transforms.Grayscale(1),
        # this also convert pixel value from [0,255] to [0,1]
        transforms.ToTensor(),
        # 这个还要修改
        transforms.Normalize(mean = 0.4761,
                             std = 0.1326),
    ])
    # load dataset
    # '../Label'
    val_set = GazeMe(dir_path=data_dir, part="val", transform=trans_train)
    print("validation set successfully read!")
    return val_set
