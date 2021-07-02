import torch
from ..models.dataloader import get_train_loader as trainloader
from ..models.dataloader import get_val_loader as valloader

# calculate the mean and standard deviation
def norm():
    # at this time, batch is no longer all data
    # load dataloader
    train_set, train_loader = trainloader('../../Darknet_model/label/combined_dataset', batch_size = 100,
                                             num_workers = 1,
                                             is_shuffle = True)

    val_set, val_loader = valloader('../../Darknet_model/label/combined_dataset', batch_size = 100,
                                         num_workers=1,
                                         is_shuffle=True)
    # count the number of pixels
    num_of_pixels_train = len(train_set) * 224 * 224
    num_of_pixels_val = len(val_set) * 224 * 224
    # calculate the mean value
    total_sum_train = 0
    total_sum_val = 0
    # summation in batches
    print("begin to calculate!")
    for batch in train_loader:
        total_sum_train += batch[0].sum()
    for batch in val_loader:
        total_sum_val += batch[0].sum()
    # calculate the mean
    mean_train = total_sum_train / num_of_pixels_train
    mean_val = total_sum_val / num_of_pixels_val
    # calculate the standard deviation
    sum_of_squared_error_train = 0
    sum_of_squared_error_val = 0
    # find the mean square error in batches
    for batch in train_loader:
        sum_of_squared_error_train += ((batch[0] - mean_train).pow(2)).sum()
    for batch in val_loader:
        sum_of_squared_error_val += ((batch[0] - mean_val).pow(2)).sum()
    std_train = torch.sqrt(sum_of_squared_error_train / num_of_pixels_train)
    std_val = torch.sqrt(sum_of_squared_error_val / num_of_pixels_val)
    print("end calculate!")
    return mean_train, std_train, mean_val, std_val

if __name__ == '__main__':
    mean_train, std_train, mean_val, std_val = norm()
    print(mean_train, std_train)
    print(mean_val, std_val)