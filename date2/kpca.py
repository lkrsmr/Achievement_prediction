import pandas as pd
from sklearn.decomposition import KernelPCA
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler

def rbf_kernel_pca(dataMat, gamma, n_components):
    kpca = KernelPCA(kernel='rbf', gamma=gamma, n_components=n_components)
    newMat = kpca.fit_transform(dataMat)

    # data1 = DataFrame(newMat)
    # data1.to_csv('test_KPCA.csv', index=False, header=False)
    return newMat

if __name__=='__main__':
    data1 = pd.read_csv('data.csv').values
    raw_data = data1[:, 1:11]
    print(raw_data.shape)
    X = raw_data[:, :-1]

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    new_X = rbf_kernel_pca(X, 10, 4)
    print(new_X)



