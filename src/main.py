from src.utils import *
import torch
import time
from src.preprocess import *
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
import math
from sklearn.manifold import TSNE



def train(lr=1e-4, train_set=None, test_set=None):
    model = LeNet28(1)
    model.cuda()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)  #数据加载器,每次抛出64条
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)
    optimizer = optim.Adam(model.parameters(), lr=lr)  #优化器
    criterion = CrossEntropyLoss()  #交叉熵
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)  # 变学习率
    step = -1
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    for epoch in range(0, 3):
        # train phase

        model.train(True)  # Set model to training mode

        train_corrects = 0.
        train_loss = 0.

        for batch_cnt, data in enumerate(train_loader):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            step += 1
            model.train(True)
            # print data
            startTime = time.time()
            inputs, labels = data
            inputs = inputs.type(torch.FloatTensor)
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.long().cuda())
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs, last_layer = model(inputs)  ##############################



            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            # print('preds', preds)
            # print('labels:', labels)
            loss.backward()
            optimizer.step()
            # print('correct_num:', torch.sum((preds == labels)).item())
            train_corrects += torch.sum((preds == labels)).item()
            train_loss += loss.item()

            # batch loss
            if step % 10 == 0:
                _, preds = torch.max(outputs, 1)

                batch_corrects = torch.sum((preds == labels)).item()
                batch_acc = batch_corrects / (labels.size(0))

                print('[%d-%d], batch_corrects: %f, batch_acc: %f' % (epoch, batch_cnt, batch_corrects, batch_acc))

        train_acc = train_corrects / (len(train_set))
        print('epoch_%d_train_acc: %f' % (epoch, train_acc))
        train_loss = math.log(train_loss)
        print(train_loss)
        train_loss_list.append(train_loss)
        exp_lr_scheduler.step()

        endTime = time.time()
        print('train time:', endTime - startTime)
        if epoch == 49:
            torch.save(model.state_dict(), './%d_params.pth' %epoch)



        # 开始测试
        # if epoch % 5 == 0:
        model.train(False)
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
            # print('preds', preds)
            # print('labels:', labels)
            # print('test_correct_num:', torch.sum((preds == labels)).item())
            test_corrects += torch.sum((preds == labels)).item()
            test_loss += loss.item()

            # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            # plot_only = 500
            # low_dim_embs = tsne.fit_transform(last_layer.cpu().data.numpy()[:plot_only, :])
            # low_dim_labels = labels.cpu().numpy()[:plot_only]
            # plot_with_labels(low_dim_embs, low_dim_labels)

        test_acc = test_corrects / (len(test_set))
        print('epoch_%d_test_acc: %f' % (epoch, test_acc))

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    return train_acc_list, test_acc_list, train_loss_list


def main():
    path = r'\data\0HP'
    # train_X, train_Y, valid_X, valid_Y, test_X, test_Y = prepro(d_path=path,
    #                                                             length=784,
    #                                                             number=1000,
    #                                                             normal=False,
    #                                                             rate=[0.7, 0.3],
    #                                                             enc=False,
    #                                                             enc_step=28)
    train_X, train_Y, test_X, test_Y = prepro(d_path=path,      length=784,
                                                                number=1000,
                                                                normal=True,
                                                                rate=[0.7, 0.3],
                                                                enc=False,
                                                                enc_step=28)

    print(test_X.shape, train_Y.shape)
    train_X = train_X.reshape(7000, 1, 28, 28)
    test_X = test_X.reshape(3000, 1, 28, 28)
    train_set = dataset(train_X, train_Y)
    test_set = dataset(test_X, test_Y)
    # print(test_set.shape)

    lr = 0.001

    # train_acc, test_acc, train_loss= train(lr, train_set, test_set)

    # train(lr, train_set, test_set)############################
    get_tsne_data(test_set)
    # plt_tsne()

    # text_save(r'C:\Users\asus\Desktop\现代信号处理大作业3\torch\src\train_acc_noN.txt', train_acc)
    # plot_acc(train_acc, train_loss)


if __name__ == '__main__':
    main()
