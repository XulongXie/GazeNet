import matplotlib.pyplot as plt
import numpy as np
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

#--------------------- first ---------------------#
# training
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
# path can be changed according to the request
plt.savefig('./first_dataset/Train_loss.jpg', dpi = 350)
plt.figure(2)
plt.plot(train_epoch, train_accurancy, 'r-', label = u'Train Accuracy')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"Accurancy", font1)
plt.title("Train Accuracy", font1)
# path can be changed according to the request
plt.savefig('./first_dataset/Train_accurancy.jpg', dpi = 350)

# validation
first_val_loss = np.loadtxt("../logs/first_dataset/val_loss.txt")
val_epoch = first_val_loss[:, 0]
val_loss = first_val_loss[:, 1]
val_L2 = first_val_loss[:, 2]

plt.figure(3)
plt.plot(val_epoch, val_loss, 'bo-', label = u'Validation loss')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"loss", font1)
plt.title("Validation loss", font1)
# path can be changed according to the request
plt.savefig('./first_dataset/Val_loss.jpg', dpi = 350)
plt.figure(4)
plt.plot(val_epoch, val_L2, 'r*-', label = u'Validation L2')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"L2", font1)
plt.title("Validation L2", font1)
# path can be changed according to the request
plt.savefig('./first_dataset/Val_L2.jpg', dpi = 350)

# accuracy on validation set
first_val = np.loadtxt("../logs/first_dataset/val.txt")
val_epoch = first_val[:, 0]
val_accurancy = first_val[:, 1]
plt.figure(5)
plt.plot(val_epoch, val_accurancy, 'r-', label = u'Validation Accuracy')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"Accurancy", font1)
plt.title("Validation Accuracy", font1)
# path can be changed according to the request
plt.savefig('./first_dataset/Val_accurancy.jpg', dpi = 350)

#--------------------- second ---------------------#
# MyGaze B training results
second_train = np.loadtxt("../logs/second_dataset/train.txt")
train_epoch = second_train[:, 0]
train_loss = second_train[:, 1] / 811
train_accurancy = second_train[:, 2]

plt.figure(6)
plt.plot(train_epoch, train_loss, 'b-', label = u'Train loss')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"loss", font1)
plt.title("Train loss", font1)
# path can be changed according to the request
plt.savefig('./second_dataset/Train_loss.jpg', dpi = 350)
plt.figure(7)
plt.plot(train_epoch, train_accurancy, 'r-', label = u'Train Accuracy')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"Accurancy", font1)
plt.title("Train Accuracy", font1)
# path can be changed according to the request
plt.savefig('./second_dataset/Train_Accurancy.jpg', dpi = 350)

# validation
second_val_loss = np.loadtxt("../logs/second_dataset/val_loss.txt")
val_epoch = second_val_loss[:, 0]
val_loss = second_val_loss[:, 1]
val_L2 = second_val_loss[:, 2]

plt.figure(8)
plt.plot(val_epoch, val_loss, 'bo-', label = u'Validation loss')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"loss", font1)
plt.title("Validation loss", font1)
# path can be changed according to the request
plt.savefig('./second_dataset/Val_loss.jpg', dpi = 350)
plt.figure(9)
plt.plot(val_epoch, val_L2, 'r*-', label = u'Validation L2')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"L2", font1)
plt.title("Validation L2", font1)
# path can be changed according to the request
plt.savefig('./second_dataset/Val_L2.jpg', dpi = 350)

# accuracy on validation set
second_val = np.loadtxt("../logs/second_dataset/val.txt")
val_epoch = second_val[:, 0]
val_accurancy = second_val[:, 1]
plt.figure(10)
plt.plot(val_epoch, val_accurancy, 'r-', label = u'Validation Accuracy')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"Accurancy", font1)
plt.title("Validation Accuracy", font1)
# path can be changed according to the request
plt.savefig('./second_dataset/Val_accurancy.jpg', dpi = 350)

#--------------------- combined ---------------------#
# training
comb_train = np.loadtxt("../logs/combined_dataset/train.txt")

train_epoch = comb_train[:, 0]
train_loss = comb_train[:, 1] / (361 + 811)
train_accurancy = comb_train[:, 2]

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

plt.figure(11)
plt.plot(train_epoch, train_loss, 'b-', label = u'Train loss')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"loss", font1)
plt.title("Train loss", font1)
# path can be changed according to the request
plt.savefig('./combined_dataset/Train_loss.jpg', dpi = 350)
plt.figure(12)
plt.plot(train_epoch, train_accurancy, 'r-', label = u'Train Accuracy')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"Accurancy", font1)
plt.title("Train Accuracy", font1)
# path can be changed according to the request
plt.savefig('./combined_dataset/Train_accurancy.jpg', dpi = 350)

# validation
comb_val_loss = np.loadtxt("../logs/combined_dataset/val_loss.txt")
val_epoch = comb_val_loss[:, 0]
val_loss = comb_val_loss[:, 1]
val_L2 = comb_val_loss[:, 2]

plt.figure(13)
plt.plot(val_epoch, val_loss, 'bo-', label = u'Validation loss')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"loss", font1)
plt.title("Validation loss", font1)
# path can be changed according to the request
plt.savefig('./combined_dataset/Val_loss.jpg', dpi = 350)
plt.figure(14)
plt.plot(val_epoch, val_L2, 'r*-', label = u'Validation L2')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"L2", font1)
plt.title("Validation L2", font1)
# path can be changed according to the request
plt.savefig('./combined_dataset/Val_L2.jpg', dpi = 350)

# accuracy on validation set
comb_val = np.loadtxt("../logs/combined_dataset/val.txt")
val_epoch = comb_val[:, 0]
val_accurancy = comb_val[:, 1]
plt.figure(15)
plt.plot(val_epoch, val_accurancy, 'r-', label = u'Validation Accuracy')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"Accurancy", font1)
plt.title("Validation Accuracy", font1)
# path can be changed according to the request
plt.savefig('./combined_dataset/Val_accurancy.jpg', dpi = 350)
plt.show()
