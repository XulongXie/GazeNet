import shutil, os, time, argparse
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data

from GazeNet_end2end.models.gazeNet import GazeNetModel as Network
from GazeNet_end2end.models.gazeData import get_train_set, get_val_set

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR


# return a boolean value for usage
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--sink', type = str2bool, nargs = '?', const = True, default = False, help = "Just sink and terminate.")
parser.add_argument('--reset', type = str2bool, nargs = '?', const = True, default = False,
                    help = "Start from scratch (do not load).")
args = parser.parse_args()

# Change there flags to control what happens.
doLoad = False  # Load checkpoint at the beginning
doTest = False  # Only run test, no training

workers = 4
epochs = 50
# Change if out of cuda memory
batch_size = 16

initial_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
lr = initial_lr

Cosine_lr = True

count_test = 0
count = 0
best_loss = 1e20

class runManager():
    def __init__(self):

        # record the parameters of each epoch
        self.epoch_count = 0                # number of epochs
        self.epoch_loss = 0                 # loss per epoch
        self.epoch_num_correct = 0          # the correct prediction for each epoch
        self.epoch_start_time = None        # epoch start time

        # record each run (different hyperparameter background)
        self.run_count = 0                  # the first run is related to batch_size
        self.run_start_time = None          # start time of each run
        self.loader = None                  # data

    # The operations that need to be performed at the beginning of each running 
    # need to pass in a network and data and necessary hyper-parameters, and put them in RunBilder for management
    def begin_run(self, loader):
        # start time
        self.run_start_time = time.time()
        # record the number of runs
        self.run_count += 1
        self.loader = loader

    # what needs to be done at the end of each running
    def end_run(self):
        # reset the number of epochs to zero
        self.epoch_count = 0

    # what needs to be done at the beginning of each epoch
    def begin_epoch(self):
        # record start time
        self.epoch_start_time = time.time()
        # record the number of epochs
        self.epoch_count += 1
        # reset the epoch loss to zero
        self.epoch_loss = 0
        # reset the correct number of epochs to zero
        self.epoch_num_correct = 0

    # what needs to be done at the end of each epoch
    def end_epoch(self):
        # calculate the time it takes to complete each epoch
        epoch_duration = time.time() - self.epoch_start_time
        # calculate the time used for each running (all epochs), 
        # here you need to pay attention, here is actually accumulating the epoch time
        run_duration = time.time() - self.run_start_time
        # calculate the correct rate
        loss = self.epoch_loss
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        # Visualize the results in the form of a table, each epoch is the smallest unit, so it should be visualized here
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        print('runs: ' + "%d" % results["run"] + ', ' + 'epoch: ' + "%d" % results["epoch"] + ', ' +
              'loss: ' + "%d" % results["loss"] + ', ' + 'accuracy: ' + "%f" % results["accuracy"])

    # the method of calculating loss, batch[0].shape[0] is actually batch_size
    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    # method of calculating the correct number
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim = 1).eq(labels).sum().item()


def main():
    global args, best_loss, weight_decay, momentum

    net = Network()

    epoch = 0
    saved = load_checkpoint()

    # can be changed with request
    dataTrain = get_train_set(data_dir = '../label/first_dataset')
    dataVal = get_val_set(data_dir = '../label/first_dataset')

    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size = batch_size, shuffle = True,
        num_workers = workers, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size = batch_size, shuffle = False,
        num_workers = workers, pin_memory = True)

    # if the effect is okay, you can consider removing weight_decay and try again
    optimizer = torch.optim.SGD(net.parameters(), lr,
                                momentum = momentum,
                                weight_decay = weight_decay)
    # cosine annealing
    if Cosine_lr:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5)
    else:
        lr_scheduler = StepLR(optimizer, step_size = 1, gamma = 0.92)

    if doLoad:
        if saved:
            print('Loading checkpoint for epoch %05d ...' % (saved['epoch']))
            state = saved['model_state']
            try:
                net.module.load_state_dict(state)
            except:
                net.load_state_dict(state)
            epoch = saved['epoch']
            best_loss = saved['best_loss']
            optimizer = saved['optim_state']
            lr_scheduler = saved['scheule_state']
        else:
            print('Warning: Could not read checkpoint!')

    # Quick test
    if doTest:
        validate(val_loader, net, epoch)
        return

    '''
    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
    '''

    m.begin_run(train_loader)
    n.begin_run(val_loader)
    print("start to run!")
    for epoch in range(epoch, epochs):

        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, net, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        loss_total = validate(val_loader, net, epoch)

        # remember best loss and save checkpoint
        is_best = loss_total < best_loss
        best_loss = min(loss_total, best_loss)
        state = {'epoch': epoch + 1,
                 'model_state': net.state_dict(),
                 'optim_state': optimizer.state_dict(),
                 'scheule_state': lr_scheduler.state_dict(),
                 'best_loss': best_loss,
                 }
        save_checkpoint(state, is_best, epoch)
    # end a running
    n.end_run()
    m.end_run()


def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()
    m.begin_epoch()
    for i, batch in enumerate(train_loader):
        img, labels = batch

        img = torch.autograd.Variable(img, requires_grad = True)
        labels = torch.autograd.Variable(labels, requires_grad=False)

        # compute output
        preds = model(img)
        # compute loss
        loss = F.cross_entropy(preds, labels)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute loss and count of correct prediction
        m.track_loss(loss, batch)
        m.track_num_correct(preds, labels)

        # print('Epoch (train): [{0}][{1}/{2}]\t'.format(epoch + 1, i, len(train_loader)))
    # end a epoch
    m.end_epoch()


def validate(val_loader, model, epoch):
    # switch to evaluate mode
    batch_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()
    model.eval()
    n.begin_epoch()
    end = time.time()
    for i, batch in enumerate(val_loader):
        img, labels = batch

        img = torch.autograd.Variable(img, requires_grad = False)
        labels = torch.autograd.Variable(labels, requires_grad = False)

        # compute output
        with torch.no_grad():
            preds = model(img)

        loss = F.cross_entropy(preds, labels)
        pred = preds.argmax(dim = 1)
        lossLin = pred - labels
        lossLin = torch.mul(lossLin, lossLin)
        lossLin = torch.sum(lossLin, 0)
        # lossLin = torch.mean(torch.sqrt(torch.tensor(lossLin,dtype = torch.float)))
        lossLin = torch.mean(torch.sqrt(lossLin.float()))

        losses.update(loss.data.item(), img.size(0))
        lossesLin.update(lossLin.item(), img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        n.track_loss(loss, batch)
        n.track_num_correct(preds, labels)

        print('Epoch (val): [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
            epoch + 1, i, len(val_loader), batch_time = batch_time,
            loss = losses, lossLin = lossesLin))

    n.end_epoch()
    print('-----------------------------------------')
    return losses.avg


# This can be changed according to the path you need
CHECKPOINTS_PATH = '.'

def load_checkpoint(filename = 'checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state


def save_checkpoint(state, is_best, epoch, filename = 'checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)
        print('Best epoch is: {0}' .format(epoch + 1))

# Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# own training strategy
def adjust_learning_rate(optimizer, epoch):
    # set the learning rate to the initial LR decayed by 10 every 30 epochs
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    # instantiate a RunManager
    m = runManager()
    n = runManager()
    main()
    print('DONE')
