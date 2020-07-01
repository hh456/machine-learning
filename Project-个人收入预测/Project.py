import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing



#加载文件
def loadfile():
    file = open('题目4 个人收入预测.csv')
    x_train = []#训练集数据
    y_train = []#训练集标签
    x_test = []#测试集数据
    y_test = []#测试集标签
    for line in file:
        line = line.strip().split(',')
        if int(line[0]) <= 3000:#取前3000行作为训练集
            line_x = list(map(float, line[1:58]))
            line_x.append(1.0)#增加一个维度
            x_train.append(line_x)
            y_train.append(float(line[58]))#取标签值
        else:
            line_x = list(map(float, line[1:58]))
            line_x.append(1.0)
            x_test.append(line_x)
            y_test.append(float(line[58]))
    x_train = np.mat(x_train)
    y_train = np.mat(y_train)
    x_test = np.mat(x_test)
    y_test = np.mat(y_test)
    return x_train,y_train,x_test,y_test

#求损失函数的值
def lossfunc(x_train,y_train,w):
    h=sigmoid(x_train * w)
    y_train = np.array(y_train)[0]
    loss = -(((y_train * np.array(np.log(h+0.000001).T))[0]+(1-y_train)*np.array(np.log(1-h+0.000001).T) )/np.shape(x_train)[0])
    return sum(loss[0])

def sigmoid(z):
    h=1.0/(1+np.exp(-z))
    return h

#梯度下降法进行训练
def train(learningrate,iteration,x_train,y_train):
    m = np.shape(x_train)[0]
    w = np.zeros((58, 1))#给w初始化为0
    loss = []
    L=2
    decay=0.25
    for k in range(iteration):
        learningrate=learningrate*decay**(k/iteration)
        h = sigmoid(x_train * w)  #sigmoid函数
        grad=np.transpose(x_train) * (h - y_train.T)/m#求梯度
        w = (1-L*learningrate/m)*w - learningrate*grad#梯度下降法公式,L2正则化
        loss.append(lossfunc(x_train, y_train, w))#损失函数
    return w.T,loss

#测试
def test(x_test,y_test,w):
    num_right = 0#记录测试正确数
    for i in range(np.shape(x_test)[0]):
        x = np.array(x_test[i])#取一组数据
        x = x[0]
        h = sigmoid(x*w.T)#通过sigmoid函数计算测试值
        if int(h+0.5)/1 == y_test[0,i]:#将测试值与实际值对比，并记录测试正确的此时
            num_right += 1
    return num_right/np.shape(x_test)[0]#返回准确率
#归一化处理
def z_score_normalization(x):
    for i in range(np.shape(x)[1]):
        std = np.std(x[:,i])#求方差
        if std == 0.0 :
            x[:,i] = 1
        else:
            x[:, i] = preprocessing.scale(x[:, i])#通过sklearn库进行归一化
    return x#返回归一化后的数据集

def min_max_normalization(x):
    for i in range(np.shape(x)[1]):
        min = np.min(x[:,i])#最小值
        max = np.max(x[:,i])#最大值
        if max == min:
            x[:, i] = 0.5
        else:
            x[:, i] = (x[:, i] - min) / (max - min)#最小值最大值归一化
    return x#返回归一化后的结果



if __name__ == '__main__':

    x_train,y_train,x_test,y_test = loadfile()
    x_train = z_score_normalization(x_train)
    x_test = z_score_normalization(x_test)
#    x_train = min_max_normalization(x_train)
#    x_test = min_max_normalization(x_test)

    learningrate = 7.5
    iteration = 1000
    w, loss1 = train(learningrate, iteration, x_train, y_train)
    rate = test(x_test, y_test, w)
    print('学习率:',learningrate,'，迭代次数:',iteration,'，正确率:', rate)
    learningrate = 1
    w, loss2 = train(learningrate, iteration, x_train, y_train)
    rate = test(x_test, y_test, w)
    print('学习率:', learningrate, '，迭代次数:', iteration, '，正确率:', rate)
    learningrate = 10
    w, loss3 = train(learningrate, iteration, x_train, y_train)
    rate = test(x_test, y_test, w)
    print('学习率:', learningrate, '，迭代次数:', iteration, '，正确率:', rate)
    learningrate = 5
    w, loss4 = train(learningrate, iteration, x_train, y_train)
    rate = test(x_test, y_test, w)
    print('学习率:', learningrate, '，迭代次数:', iteration, '，正确率:', rate)



    kl = [i for i in range(iteration)]
    plt.plot(kl, loss1, 'b-')
    plt.plot(kl, loss2, 'r-')
    plt.plot(kl, loss3, 'g-')
    plt.plot(kl, loss4, 'y-')
    plt.show()





