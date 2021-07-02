import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
from PIL import Image
import matplotlib.pyplot as plt


class GazeMe(Dataset):
    # __init__ method will read data file
    def __init__(self, dir_path, part, transform=None):
        # initialize the properties inherited from the parent class
        super(GazeMe, self).__init__()
        # absolute path of all pictures
        assert part in ["train", "val", "test"]
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


def get_train_loader(data_dir, batch_size, num_workers, is_shuffle):
    # load dataset
    # '../Label'
    train_set = GazeMe(dir_path = data_dir, part = "test",
                       transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()]))
    # create a dataloader
    train_loader = DataLoader(train_set, batch_size = batch_size, num_workers = num_workers, shuffle=is_shuffle)

    return train_set, train_loader

def get_val_loader(data_dir, batch_size, num_workers, is_shuffle):
    # load dataset
    # '../Label'
    val_set = GazeMe(dir_path = data_dir, part = "test",
                       transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()]))
    # create a dataloader
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=is_shuffle)

    return val_set, val_loader


def PIL_to_tensor(image):
    loader = transforms.Compose([
        transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image


def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    # remove the fake batch dimension
    image = image.squeeze(0)
    image = unloader(image)
    return image


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    # we clone the tensor to not do changes on it
    image = tensor.cpu().clone()
    # remove the fake batch dimension
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    # pause a bit so that plots are updated
    plt.pause(0.001)


