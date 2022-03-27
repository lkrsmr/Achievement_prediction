#数据预处理
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import resample
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.02 * height, '%s' % int(height))

#data augmentation

def stretch(x):
    l = len(x)
    y = resample(x, l)#对x进行重采样，采样数量为L
    y_ = y[:8]#如果len(y)>271，那么只取前271个值
    return y_

def amplify(x, alpha):
    factor = -alpha*x + (1+alpha)
    F = x*factor #f=-ax^2+(1+a)x来生成新数据
    return F

def augment(x, l):
    result = np.zeros(shape=(l, 8))
    alpha = random.uniform(0.5, 1) #随机取（0.5,1）之间的一个数
    for i in range(l):#每次造l个
        new_y = stretch(x)
        new_y = amplify(new_y, alpha)
        result[i, :] = new_y
    return result

def Data_expansion(count, X):
    l = count//X.shape[0]-1
    mod = count%X.shape[0]
    res = []
    for i in range(len(X)):
        res.append(augment(X[i], l))
    res = np.array(res)
    res = res.reshape(-1, 8)
    ans = []
    for i in range(mod):
        ans.append(augment(X[i], 1))
    ans = np.array(ans)
    #print(len(res))
    ans = ans.reshape(-1, 8)
    new_data = np.vstack([res, X, ans])
    return new_data, len(res)

def data_main():
    data1 = pd.read_csv('data.csv').values
    raw_data = data1[:, 1:11]
    print(raw_data.shape)

    X = raw_data[:, :-1]
    y = raw_data[:, -1].astype(int)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    y_pic = pd.Series(y)
    #df.values[:, -1] = df.values[:, -1].astype(int)
    equilibre = y_pic.value_counts()
    print(equilibre)
    plt.figure(figsize=(6, 6), dpi=100)
    my_circle = plt.Circle((0, 0), 0.75, color='white')
    plt.pie(equilibre, labels=['0', '1', '2'], colors=['red', 'green', 'blue'],
            autopct='%1.2f%%')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()

    print(X.shape)
    print(y.shape)

    C0 = np.argwhere(y == 0).flatten()
    C1 = np.argwhere(y == 1).flatten()
    C2 = np.argwhere(y == 2).flatten()

    l1 = [len(C0), len(C1), len(C2)]
    total_width, n = 0.8, 1
    width = total_width / n
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    x1 = ['0', '1', '2']
    a = plt.bar(x1, l1, width=width)
    autolabel(a)
    plt.xlabel('类别标签')
    plt.ylabel('数量')
    plt.grid(ls='--')
    plt.show()

    class_0, k0 = Data_expansion(8894, X[C0])
    class_1 = X[C1]
    class_2, k2 = Data_expansion(8894, X[C2])

    # plt.figure(figsize=(8, 6), dpi=100)
    # plt.subplot(311)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.plot(class_1[0], 'r--', label='新样本')
    # plt.plot(class_1[k1], 'g--', label='原始样本')
    # plt.legend()
    # plt.title('S 类')
    # plt.grid(ls='--')
    #
    # plt.subplot(312)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.plot(class_2[1], 'r--', label='新样本')
    # plt.plot(class_2[k2+1], 'g--', label='原始样本')
    # plt.title('V 类')
    # plt.legend()
    # plt.grid(ls='--')
    #
    # plt.subplot(313)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.plot(class_3[80], 'r--', label='新样本')
    # plt.plot(class_3[k3+80], 'g--', label='原始样本')
    # plt.title('F 类')
    # plt.xlabel('采样点')
    # plt.legend()
    # plt.grid(ls='--')
    # plt.tight_layout()
    # plt.show()

    y_0 = np.zeros(shape=(class_0.shape[0],), dtype=int)
    y_1 = np.ones(shape=(class_1.shape[0],), dtype=int)*1
    y_2 = np.ones(shape=(class_2.shape[0],), dtype=int)*2


    new_X = np.vstack([class_0, class_1, class_2])
    new_y = np.hstack([y_0, y_1, y_2])
    random_seed = 2
    train_X, test_X, train_Y, test_Y = train_test_split(new_X, new_y, test_size=0.1, random_state=random_seed)
    return train_X, train_Y, test_X, test_Y

if __name__=='__main__':
    train_X, train_Y, test_X, test_Y = data_main()
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
