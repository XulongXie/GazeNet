from GazeNet_model.models.gazeData import GazeMe
from GazeNet_model.models.gazeNet import GazeNetModel as Network

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim
import torch.utils.data

import time, os
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

doLoad = True   # Load checkpoint at the beginning
workers = 4
epochs = 100
# Change if out of cuda memory
batch_size = 16

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

    net = Network()
    saved = load_checkpoint()
    # can be changed accoding to request
    dataEval = get_eval_set(data_dir = '../label/second_dataset')

    eval_loader = torch.utils.data.DataLoader(
        dataEval,
        batch_size = batch_size, shuffle = False,
        num_workers = workers, pin_memory = True)

    if doLoad:
        if saved:
            print('Loading checkpoint for epoch %05d ...' % (saved['epoch']))
            state = saved['model_state']
            try:
                net.module.load_state_dict(state)
            except:
                net.load_state_dict(state)
            epoch = saved['epoch']
        else:
            print('Warning: Could not read checkpoint!')


    m.begin_run(eval_loader)
    print("start to run!")

    pred_set = evaluation(eval_loader, net, epoch)
    drawConfusionMatrix(pred_set, dataEval.targets, dataEval.classes)

    # end a running
    m.end_run()


def evaluation(eval_loader, model, epoch):
    # switch to evaluate mode
    batch_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()
    # create a tensor to store all prediction results
    preds_set = torch.tensor([])
    model.eval()
    m.begin_epoch()
    end = time.time()
    for i, batch in enumerate(eval_loader):
        imEyeL, imEyeR, EyeCorner, gaze = batch

        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
        EyeCorner = torch.autograd.Variable(EyeCorner.float(), requires_grad = False)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        with torch.no_grad():
            preds = model(imEyeL, imEyeR, EyeCorner)
            preds_set = torch.cat((preds_set, preds), dim = 0)

        loss = F.cross_entropy(preds, gaze)
        pred = preds.argmax(dim = 1)
        lossLin = pred - gaze
        lossLin = torch.mul(lossLin, lossLin)
        lossLin = torch.sum(lossLin, 0)
        # lossLin = torch.mean(torch.sqrt(torch.tensor(lossLin,dtype = torch.float)))
        lossLin = torch.mean(torch.sqrt(lossLin.float()))

        losses.update(loss.data.item(), imEyeL.size(0))
        lossesLin.update(lossLin.item(), imEyeL.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        m.track_loss(loss, batch)
        m.track_num_correct(preds, gaze)

        print('Epoch (val): [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
            epoch + 1, i, len(eval_loader), batch_time = batch_time,
            loss = losses, lossLin = lossesLin))

    m.end_epoch()
    return preds_set

# this can be changed according to the path you need
CHECKPOINTS_PATH = '../logs/second_dataset'

def load_checkpoint(filename = 'Epoch3_best_checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

def get_eval_set(data_dir):
    trans_eval = transforms.Compose([
        transforms.Grayscale(1),
        # this also convert pixel value from [0,255] to [0,1]
        transforms.ToTensor(),
        # First dataset: 0.4643, 0.1649
        # Second dataset: 0.4761, 0.1326
        # Combined dataset: 0.4715, 0.1450
        transforms.Normalize(mean = 0.4761,
                             std = 0.1326),
    ])
    # load dataset
    eval_set = GazeMe(dir_path = data_dir, part = "val", transform = trans_eval)
    print("evaluation set successfully read!")
    return eval_set

# Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = "center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # can be changed according to request
    plt.savefig('../analysis/second_dataset/confusion.jpg', dpi=350)

def drawConfusionMatrix(preds_set, labels, classes):
    Confusion_Matrix = confusion_matrix(labels, preds_set.argmax(dim = 1))
    print(Confusion_Matrix)
    plt.figure(0)
    plot_confusion_matrix(Confusion_Matrix, classes)
    plt.show()

if __name__ == "__main__":
    # instantiate a RunManager
    m = runManager()
    main()
    print('DONE')