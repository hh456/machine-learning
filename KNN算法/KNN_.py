import numpy as np


def load_file(train_num,test_num):

    train_image=np.zeros((train_num,28*28))
    test_image=np.zeros((test_num,28*28))
    train_label = np.zeros(train_num)
    test_label = np.zeros(test_num)
    file_data=open('train-images.IDX3-UBYTE','rb')
    file_data.seek(16)
    file_label=open('train-labels.idx1-ubyte','rb')
    file_label.seek(8)
    for i in range(train_num):
        for j in range(28 * 28):
            data = file_data.read(1)
            pixel = int.from_bytes(data, byteorder='big')
            if pixel > 10:
                train_image[i][j] = 1

        data = file_label.read(1)
        sign = int.from_bytes(data, byteorder='big')
        train_label[i] = sign

    file_data2=open('test-images.IDX3-UBYTE','rb')
    file_data2.seek(16)
    file_label2=open('test-labels.idx1-ubyte','rb')
    file_label2.seek(8)
    for i in range(test_num):
        for j in range(28 * 28):
            data = file_data2.read(1)
            pixel = int.from_bytes(data, byteorder='big')
            if pixel > 10:
                test_image[i][j] = 1

        data = file_label2.read(1)
        sign = int.from_bytes(data, byteorder='big')
        test_label[i] = sign

    return train_image,test_image,train_label,test_label

def knn(test_image,train_image,train_label,k):
    lower_k=1
    while (lower_k==1)and(k>0):
        all_distance = (np.sum((np.tile(test_image, (train_image.shape[0], 1)) - train_image) ** 2,axis=1)) ** 0.5
        order=np.zeros((10),dtype=int)
        sort_dis=all_distance.argsort()
        for i in range(k):
            vote_label=train_label[sort_dis[i]]
            order[int(vote_label)] += 1
        max_time=0
        result=-1
        for i in range(10):
            if order[i]>max_time:
                max_time=order[i]
                result=i
                lower_k=0
            if order[i]==max_time:
                lower_k=1
        k=k-1
    return result

def test(k,test_image,train_image,train_label):
    right=0.0
    for i in range(test_image.shape[0]):
        label = knn(test_image[i], train_image, train_label, k)
        if test_label[i] == label:
            right = right + 1
    rate = right / test_image.shape[0]
    return rate

if __name__ == '__main__':

    train_num=5000
    test_num=500
    k=4
    train_image,test_image,train_label,test_label=load_file(train_num,test_num)
    rate=test(k,test_image,train_image,train_label)
    print('正确率：',rate)



