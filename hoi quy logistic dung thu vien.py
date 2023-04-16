
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# lưu weight dùng numpy.save(), định dạng '.npy'
from sklearn.linear_model import LogisticRegression
w = np.array([0., 0.1, 0.1]).reshape(-1, 1)
np.save('weight logistic.npy', w)
# Load weight từ file '.npy'
w = np.load('weight logistic.npy')

# Logistic Regression dùng thư viện sklearn

# load data tu file csv
data = pd.read_csv('dataset.csv').values
N, d = data.shape
x = data[:, :d - 1].reshape(-1, d - 1)
y = data[:, 2].reshape(-1, 1)

# ve data bang scatter
plt.scatter(x[:10, 0], x[:10, 1], c='red', edgecolors='pink', label='cho vay')
plt.scatter(x[10:, 0], x[10:, 1], c='blue', edgecolor='pink', label='tu choi')
plt.legend(loc=1);
plt.xlabel('muc lương (trieu)')
plt.ylabel('kinh nghiem (nam)')

#tạo model logistic regression và train
logreg = LogisticRegression()
logreg.fit(x, y)

#lưu các biến của mô hình vào mảng
wg = np.zeros((3, 1))
wg[0,0] = logreg.intercept_
wg[1:,0] = logreg.coef_

#vẽ đường thẳng phân cách          wg la mang 1 chieu khong phai 1 so dang loi khong biet fix
t = 0,5
plt.plot((4,10),( -(wg[0] + 4*wg[1] + np.log(1/t - 1))/wg[2],
                -(wg[0] + 10*wg[1] + np.log(1/t - 1))/wg[2]), 'o:g')
plt.sh()
np.savez('w logistic.npz', a=logreg.intercept_, b=logreg.coef_)
# Load các tham số dùng numpy.load(), file '.npz'
k = np.load('w logistic.npz')
logreg.intercept_ = k['a']
logreg.coef_ = k['b']
