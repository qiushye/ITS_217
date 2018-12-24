import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import sys

###############################################################################
# Generate sample data
X = np.sort(
    5 * np.random.rand(40, 2), axis=0)  #产生40组数据，每组一个数据，axis=0决定按列排列，=1表示行排列
y = np.random.rand(40)

###############################################################################
# Add noise to targets
# y[::5] += 3 * (0.5 - np.random.rand(8))
print(X.shape, y.shape)
# sys.exit()
###############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  #rbf的效果是最好的
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)
