from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from xgboost import XGBClassifier
from keras.layers import Dense, Dropout
from keras.models import Input, Model
import warnings
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from kpca import rbf_kernel_pca

warnings.filterwarnings("ignore")

def xgboost_model():
    model = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=10, silent=True)
    return model

def svm1_model():
    model = SVC(kernel='rbf', gamma='auto')
    return model

def knn_model():
    model = KNeighborsClassifier(n_neighbors=3)
    return model

def dt_model():
    model = DecisionTreeClassifier(max_depth=5)
    return model

def rf_model():
    model = RandomForestClassifier(n_estimators=10, max_depth=5)
    return model

def mlp_model(number_features, output_node):
  input_size = (number_features, )
  inputs = Input(shape=input_size)
  d1 = Dense(128, activation='relu')(inputs)
  D1 = Dropout(0.1)(d1)
  d2 = Dense(64, activation='relu')(D1)
  D2 = Dropout(0.1)(d2)
  outputs = Dense(output_node, activation='softmax')(D2)
  model = Model(inputs=inputs, outputs=outputs)
  return model

def bayes_model():
    model = GaussianNB()
    return model

def evaluate(actual, predicted):
    acc = metrics.accuracy_score(actual, predicted)
    sen = metrics.recall_score(actual, predicted, average='macro')
    #m = metrics.confusion_matrix(actual, predicted)
    return acc, sen

if __name__=='__main__':
    from sklearn.model_selection import train_test_split
    import pandas as pd

    data1 = pd.read_csv('data.csv').values
    raw_data = data1[:, 1:11]
    print(raw_data.shape)

    X = raw_data[:, :-1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    new_X = scaler.fit_transform(X)

    new_X = rbf_kernel_pca(X, 5, 5)
    y = raw_data[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(new_X, y, test_size=0.1, random_state=0)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    print("XGBoost Classifier")
    model_0 = xgboost_model()
    model_0.fit(x_train, y_train)
    predict_0 = model_0.predict(x_test)
    acc_0, sen_0 = evaluate(y_test, predict_0)
    print(acc_0, sen_0)

    print("SVM1 Classifier")
    model_1 = svm1_model()
    model_1.fit(x_train, y_train)
    predict_1 = model_1.predict(x_test)
    acc_1, sen_1 = evaluate(y_test, predict_1)
    print(acc_1, sen_1)

    print("KNN Classifier")
    model_2 = knn_model()
    model_2.fit(x_train, y_train)
    predict_2 = model_2.predict(x_test)
    acc_2, sen_2 = evaluate(y_test, predict_2)
    print(acc_2, sen_2)

    print("DT Classifier")
    model_3 = dt_model()
    model_3.fit(x_train, y_train)
    predict_3 = model_3.predict(x_test)
    acc_3, sen_3 = evaluate(y_test, predict_3)
    print(acc_3, sen_3)

    print("RF Classifier")
    model_4 = rf_model()
    model_4.fit(x_train, y_train)
    predict_4 = model_4.predict(x_test)
    acc_4, sen_4 = evaluate(y_test, predict_4)
    print(acc_4, sen_4)

    print("Bayes Classifier")
    model_7 = bayes_model()
    model_7.fit(x_train, y_train)
    predict_7 = model_7.predict(x_test)
    acc_7, sen_7 = evaluate(y_test, predict_7)
    print(acc_7, sen_7)

    print("MLP Classifier")
    y_train = to_categorical(y_train.astype(int), num_classes=3)
    y_test = to_categorical(y_test.astype(int), num_classes=3)
    opt = Adam(lr=0.001)
    model_6 = mlp_model(number_features=x_train.shape[1], output_node=3)
    model_6.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    history_6 = model_6.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), verbose=0)

    plt.plot(history_6.history['loss'], label='Train loss')
    plt.plot(history_6.history['val_loss'], label='Test loss')
    plt.legend()
    plt.show()

    plt.plot(history_6.history['acc'], label='Train Acc')
    plt.plot(history_6.history['val_acc'], label='Test Acc')
    plt.legend()
    plt.show()

    y_pred = model_6.predict(x_test, batch_size=32)
    actual = y_test.argmax(axis=1)
    predicted = y_pred.argmax(axis=1)
    acc_6, sen_6 = evaluate(actual, predicted)
    print(acc_6, sen_6)