import matplotlib.pyplot as plt
import numpy as np
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

#--------------------- first ---------------------#
# line chart of training loss and accuracy
first_train = np.loadtxt("../logs/first_dataset/train.txt")

train_epoch = first_train[:, 0]
train_loss = first_train[:, 1] / 361
train_accurancy = first_train[:, 2]

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

plt.figure(1)
plt.plot(train_epoch, train_loss, 'b-', label = u'Train loss')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"loss", font1)
plt.title("Train loss", font1)
# path can be changed according to request
plt.savefig('./first_dataset/Train_loss.jpg', dpi = 350)
plt.figure(2)
plt.plot(train_epoch, train_accurancy, 'r-', label = u'Train Accuracy')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"Accurancy", font1)
plt.title("Train Accuracy", font1)
# path can be changed according to request
plt.savefig('./first_dataset/Train_accurancy.jpg', dpi = 350)

# line chart of loss (average) and accuracy on the validation set
first_val = np.loadtxt("../logs/first_dataset/val.txt")
val_epoch = first_val[:, 0]
val_loss = first_val[:, 1] / 90
val_accurancy = first_val[:, 2]
plt.figure(3)
plt.plot(val_epoch, val_loss, 'b-', label = u'Validation loss')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"loss", font1)
plt.title("Validation loss", font1)
# path can be changed according to request
plt.savefig('./first_dataset/Val_loss.jpg', dpi = 350)
plt.figure(4)
plt.plot(val_epoch, val_accurancy, 'r-', label = u'Validation Accuracy')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"Accurancy", font1)
plt.title("Validation Accuracy", font1)
# path can be changed according to request
plt.savefig('./first_dataset/Val_accurancy.jpg', dpi = 350)

#--------------------- second ---------------------#
# results obtained from the MyGaze B data set
second_train = np.loadtxt("../logs/second_dataset/train.txt")
train_epoch = second_train[:, 0]
train_loss = second_train[:, 1] / 811
train_accurancy = second_train[:, 2]

plt.figure(5)
plt.plot(train_epoch, train_loss, 'bo-', label = u'Train loss')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"loss", font1)
plt.title("Train loss", font1)
# path can be changed according to request
plt.savefig('./second_dataset/Train_loss.jpg', dpi = 350)
plt.figure(6)
plt.plot(train_epoch, train_accurancy, 'r*-', label = u'Train Accuracy')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"Accurancy", font1)
plt.title("Train Accuracy", font1)
# path can be changed according to request
plt.savefig('./second_dataset/Train_Accurancy.jpg', dpi = 350)

# loss and accuracy of the MyGaze B validation set
second_val = np.loadtxt("../logs/second_dataset/val.txt")
val_epoch = second_val[:, 0]
val_loss = second_val[:, 1] / 144
val_accurancy = second_val[:, 2]

plt.figure(7)
plt.plot(val_epoch, val_loss, 'bo-', label = u'Validation loss')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"loss", font1)
plt.title("Validation loss", font1)
# path can be changed according to request
plt.savefig('./second_dataset/Val_loss.jpg', dpi = 350)
plt.figure(8)
plt.plot(val_epoch, val_accurancy, 'r*-', label = u'Validation Accuracy')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"Accurancy", font1)
plt.title("Validation Accuracy", font1)
# path can be changed according to request
plt.savefig('./second_dataset/Val_accurancy.jpg', dpi = 350)
plt.show()
