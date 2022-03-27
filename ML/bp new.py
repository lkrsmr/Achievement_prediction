import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

df = pd.read_excel('01test3.xlsx',index_col=0,header=0)

# 把数据集分为训练集和测试集
values = df.values

values_x, values_y = values[:, :-1], values[:, -1]
train_x,test_x,train_y,test_y = train_test_split(values_x,values_y,test_size=0.2,random_state=5)
train_y = to_categorical(num_classes=2, y =train_y)
test_y = to_categorical(num_classes=2, y=test_y)

# 设计网络
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


reduce_lr = LearningRateScheduler(scheduler)
model = Sequential()
model.add(Dense(17, input_shape=(8,), activation='relu'))#
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['acc'])

# 拟合网络
history = model.fit(train_x,train_y,epochs=60,batch_size=20,validation_split=0.3,
                    verbose=2,shuffle=True,callbacks=[reduce_lr])


acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
# print(val_loss)
# print(loss)
import matplotlib.pyplot as plt


# 绘图
acc_epoch=[]
for i in range(len(acc)):
    acc_epoch.append(i+1)
loss_epoch=[]
for i in range(len(loss)):
    loss_epoch.append(i+1)
val_acc_epoch=[]
for i in range(len(val_acc)):
    val_acc_epoch.append(i+1)
val_loss_epoch=[]
for i in range(len(val_loss)):
    val_loss_epoch.append(i+1)

plt.rcParams['font.sans-serif']=['SimSun']
#plt.subplot(221)
plt.title('MLP-训练集和测试集的loss值')
plt.xlabel('训练次数')
plt.ylabel('loss值')
loss_line,=plt.plot(loss_epoch,loss,label="训练集上的loss",marker='o')
val_loss_line,=plt.plot(val_loss_epoch,val_loss,label="测试集上的loss",linestyle='--',marker='+')
loss_num=np.argmin(val_loss)
plt.scatter(loss_num+1,val_loss[loss_num], s=150, label='最佳训练次数')
plt.legend()
plt.show()

plt.xlabel('训练次数')
plt.ylabel('准确率')
plt.title('MLP-训练集和测试集的预测准确率')
acc_line,=plt.plot(acc_epoch,acc,label="训练集上的准确率",marker='o')
val_acc_line,=plt.plot(val_acc_epoch,val_acc,label="测试集上的准确率",linestyle='--',marker='+')
acc_num=np.argmax(val_acc)
plt.scatter(acc_num+1, val_acc[acc_num], s=150, label='最佳训练次数')
plt.legend()
plt.show()

print('防止过拟合，最佳训练次数为：',np.argmin(val_loss)+1)
print('此时训练集上损失率为：','%.2f%%'%(loss[np.argmin(val_loss)]* 100))
print('此时测试集上损失率为：','%.2f%%'%(val_loss[np.argmin(val_loss)]* 100))
print('准确率最高，最佳训练次数：',np.argmax(val_acc)+1)
print('此时训练集上预测率为：','%.2f%%'%(acc[np.argmax(val_acc)]* 100))
print('此时测试集上预测率为：','%.2f%%'%(val_acc[np.argmax(val_acc)]* 100))
scores = model.evaluate(test_x, test_y, verbose=0)
print('bp预测准确率为:', scores)
