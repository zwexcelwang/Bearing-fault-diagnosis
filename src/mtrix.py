
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import numpy as np

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

data = np.loadtxt("test_labels.txt")
pred = np.loadtxt("test_preds.txt")
cm = confusion_matrix(data, pred)
labels_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# for x in range(len(cm)):
#     for y in range(len(cm)):
#         plt.annotate(cm[x,y], xy = (x,y), horizontalalignment = 'center', verticalalignment = 'center', fontsize = 10)
plot_confusion_matrix(cm, labels_name, "Confusion Matrix")
# plt.colorbar()
plt.show()