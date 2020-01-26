import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse

def train(X, y, lr=0.001, eps=0.000001):
    w0, w1 = 0, 0
    while True:
        y_pred = w0 + w1 * X
        w0_new = w0 - lr * (y_pred - y).sum() / y.shape[0]
        w1_new = w1 - lr * np.dot(X, y_pred - y) / y.shape[0]
        if (abs(w0_new - w0) < eps and abs(w1_new - w1) < eps):
            return w0_new, w1_new
        w0, w1 = w0_new, w1_new

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', help='visualization', default=False, action='store_true')
    parser.add_argument('--precision', help='precision', default=False, action='store_true')
    parser.add_argument('-lr', '--learning_rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-eps', '--epsilon', help='epsilon', default=0.01, type=float)
    args = parser.parse_args()
    is_plot = args.plot
    is_precision = args.precision
    lr = args.learning_rate
    eps = args.epsilon

    df = pd.read_csv("data.csv")
    X, y = df['km'].values, df['price'].values

    #normalize data
    X_normalize = (X - df['km'].mean()) / df['km'].std()

    #training
    w0, w1 = train(X_normalize, y, lr, eps)

    #denormalize weights
    w0 = w0 - w1 * df['km'].mean() / df['km'].std()
    w1 = w1 / df['km'].std()

    print ("Model: y = %f + %f * x" %(w0, w1))

    with open('weights.txt', 'w') as f:
        f.write("%f %f" %(w0, w1))

    if is_precision:
        y_pred = w0 + w1 * X
        MAE = ((abs(y_pred - y)).sum() / y.shape[0])
        MSE = (((y_pred - y) ** 2).sum() / y.shape[0])
        RMSE = (((y_pred - y) ** 2).sum() / y.shape[0]) ** 0.5
        print ("Mean Absolute Error: %f" %MAE)
        print ("Mean Squared Error: %f" %MSE)
        print ("Root Mean Squared Error: %f" %RMSE)

    if is_plot:
        y_pred = w0 + w1 * X
        plt.scatter(y, X)
        plt.plot(y_pred, X, color='orange')
        plt.show()