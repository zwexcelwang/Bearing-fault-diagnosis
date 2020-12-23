import numpy as np
import matplotlib.pyplot as plt
def plot_BN(train_acc_list, test_acc_list):
    time = np.arange(0, len(train_acc_list))
    plt.plot(time, train_acc_list)
    plt.plot(time, test_acc_list)
    # plt.ylim(0,1)
    plt.legend(["Normal=False", "Normal=True"], loc=0)
    plt.savefig("BN.png")
    plt.show()
acc_BN = []
acc_noBN = []
f = open(r"train_acc_N.text","r")
data = f.readlines()
for line in data:
    line = line.strip('\n')  # 去掉列表中每一个元素的换行符
    acc_BN.append(float(line))
f = open(r"train_acc_noN.txt","r")
data = f.readlines()
for line in data:
    line = line.strip('\n')  # 去掉列表中每一个元素的换行符
    acc_noBN.append(float(line))
print(acc_noBN)
plot_BN(acc_noBN, acc_BN)
