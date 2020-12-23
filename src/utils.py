import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from matplotlib import cm


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class LeNet28(nn.Module): #for 28
    def __init__(self, input_nc):
        super(LeNet28, self).__init__()
        """nn.Conv2d(channel, outputchannel, kernel_size, stride, padding)"""
        sequence1 = [
            nn.Conv2d(input_nc, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            # nn.Dropout2d(0.5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(7 * 7 * 64, 1024),
            nn.ReLU(True),

        ]
        sequence2 = [
            # Flatten(),
            # nn.Linear(7 * 7 * 64, 1024),
            # nn.ReLU(True),
            nn.Dropout(0.8),
            nn.Linear(1024, 10),
            nn.Softmax(dim=1)
        ]

        self.net1 = nn.Sequential(*sequence1)
        self.net2 = nn.Sequential(*sequence2)

    def forward(self, input, d_feat=False):
        """Standard forward."""
        # x 是最后一层（全连接前的那一层）输出的特征
        x = self.net1(input)
        out = self.net2(x)
        return out, x



class dataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        data = self.x[item]
        label = self.y[item]
        return data, label

def plot_acc(train_acc_list, test_acc_list):
    time = np.arange(0, len(train_acc_list))
    plt.plot(time, train_acc_list)
    plt.plot(time, test_acc_list)
    plt.legend(["acc", "loss"], loc=0)
    plt.savefig("acc_loss.png")
    plt.show()

def text_save(filename, data):#filename为写入TEXT文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")

def test(test_set):
    model = LeNet28(1)
    model.cuda()
    model.load_state_dict(torch.load('./0_params.pth'))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)
    model.train(False)
    criterion = CrossEntropyLoss()
    test_corrects = 0.
    test_loss = 0.
    for batch_cnt, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.type(torch.FloatTensor)
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.long().cuda())

        outputs, last_layer = model(inputs)  ##########

        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum((preds == labels)).item()
        test_loss += loss.item()

    test_acc = test_corrects / (len(test_set))
    print('test_acc: %f' % test_acc)

def get_tsne_data(test_set):
    last = []
    label = []
    pred = []
    model = LeNet28(1)
    model.cuda()
    ####################################################################
    model.load_state_dict(torch.load('./49_0.993_0.984_params.pth'))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)
    model.train(False)
    test_corrects = 0.
    for batch_cnt, data in enumerate(test_loader):
        inputs, labels = data
        # print(labels, type(labels))
        inputs = inputs.type(torch.FloatTensor)
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.long().cuda())
        outputs, last_layer = model(inputs)  #######################
        '''需根据实际情况修改参数'''
        if last_layer.shape[0] == 64:
            last_layer = last_layer.reshape([64, -1])
            # print(last_layer.shape)
        else:
            last_layer = last_layer.reshape([56, -1])
            # print(last_layer.shape)

        last_layer = last_layer.cpu().detach().numpy().tolist()
        last = last + last_layer

        labels_ = labels.cpu().detach().numpy().tolist()
        label = label + labels_



        _, preds = torch.max(outputs, 1)
        preds_ = preds.cpu().detach().numpy().tolist()
        pred = pred + preds_

        test_corrects += torch.sum((preds == labels)).item()
    print(np.array(label).shape)
    text_save('test_data.txt', last)
    text_save('test_labels.txt', label)
    text_save('test_preds.txt', pred)

    test_acc = test_corrects / (len(test_set))
    print('test_acc: %f' % test_acc)

def plt_tsne():
    data = np.loadtxt("test_data.txt")
    # print(data.shape)
    labels = np.loadtxt("test_labels.txt")
    # print(labels)
    tsne = TSNE(n_components=2, learning_rate=200.0).fit_transform(data)
    # print(tsne)
    cm = plt.cm.get_cmap('jet')
    sc = plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap=cm)
    plt.colorbar(sc)
    plt.savefig('tSNE.png')
    plt.show()


# def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
#     labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     xlocations = np.array(range(len(labels)))
#     plt.xticks(xlocations, labels, rotation=90)
#     plt.yticks(xlocations, labels)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     cm = confusion_matrix(y_true, y_pred)
#     np.set_printoptions(precision=2)

# def plot_with_labels(lowDWeights, labels):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9));
#         plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max());
#     plt.ylim(Y.min(), Y.max());
#     plt.title('Visualize last layer');
#     plt.show();
#     plt.pause(0.01)

