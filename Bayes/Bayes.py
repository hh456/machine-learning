import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# 加载训练测试文件
def loadfile():
    ham_train = ''
    spam_train = ''
    ham_test=[]
    spam_test=[]
    for i in range(1, 36):
        f = open('ham/' + str(i) + '.txt', 'r', errors='ignore')
        ham_train = ham_train + f.read() + ' '
        f.close()
        f = open('spam/' + str(i) + '.txt', 'r', errors='ignore')
        spam_train = spam_train + f.read() + ' '
        f.close()
    ham_train = re.sub(r'\s+', ' ', re.sub('[^a-zA-Z]', ' ', ham_train)).lower()    #去除标点、符号、数字，并将大写变成小写
    spam_train = re.sub(r'\s+', ' ', re.sub('[^a-zA-Z]', ' ', spam_train)).lower()[1:]
    for i in range(36, 56):
        f = open('spam/' + str(i) + '.txt', 'r', errors='ignore')
        test = f.read()
        f.close()
        test = re.sub(r'\s+', ' ', re.sub('[^a-zA-Z]', ' ', test)).lower()
        spam_test.append(test)
    for i in range(36, 56):
        f = open('ham/' + str(i) + '.txt', 'r', errors='ignore')
        test = f.read()
        f.close()
        test = re.sub(r'\s+', ' ', re.sub('[^a-zA-Z]', ' ', test)).lower()
        ham_test.append(test)
    return ham_train,spam_train,ham_test,spam_test

# 统计垃圾邮件和健康邮件的词频
def word_num(text):
    vectorizer = CountVectorizer()
    L = ['']
    L[0] = text
    weight = vectorizer.fit_transform(L).toarray()
    word = vectorizer.get_feature_names()  # 所有文本的关键字
    return {word[j]: int(weight[0][j]) for j in range(len(word))}

# 求词频字典的总频数
def Sum(dic):
    n = 0
    for value in dic.values():
        n = n + value
    return n
#bayes计算结果
def Bayes(test,ham_dic,spam_dic):
    test_count = sorted(word_num(test).items(), key=lambda x: x[1], reverse=True)
    P = []
    for n in range(len(test_count)):
        word = test_count[n][0]
        if not spam_dic.get(word):
            P.append(0.00001)
        elif not ham_dic.get(word):
            word_ham = 0.00001
            word_spam = spam_dic[word] / Sum(spam_dic)
            P.append((word_spam * 0.5) / ((word_ham * 0.5) + (word_spam * 0.5)))
        else:
            word_spam = spam_dic[word] / Sum(spam_dic)
            word_ham = ham_dic[word] / Sum(ham_dic)
            P.append((word_spam * 0.5) / ((word_ham * 0.5) + (word_spam * 0.5)))
    p1 = 1
    p2 = 1
    for n in range(len(test_count)):
        p1 = p1 * P[n]
        p2 = p2 * (1 - P[n])
    return (p1 / (p1 + p2))
#测试求正确率
def test_bayes(ham_train,spam_train,ham_test,spam_test):
    ham_dic = dict(sorted(word_num(ham_train).items(), key=lambda x: x[1], reverse=True))
    spam_dic = dict(sorted(word_num(spam_train).items(), key=lambda x: x[1], reverse=True))
    right=0.0
    for i in range(len(spam_test)):
        if Bayes(spam_test[i],ham_dic,spam_dic) >= 0.5:
            right+=1
    for i in range(len(ham_test)):
        if Bayes(ham_test[i],ham_dic,spam_dic) < 0.5:
            right+=1
    return right/(len(ham_test)+len(spam_test))




ham_train,spam_train,ham_test,spam_test=loadfile()

rate=test_bayes(ham_train,spam_train,ham_test,spam_test)

print('正确率：',rate)

