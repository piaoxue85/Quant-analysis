import pandas as pd
import numpy as np
from timeit import default_timer as timer
from sklearn.metrics import explained_variance_score, mean_squared_error
from data.data_processor import DataForModel
pd.set_option('display.precision',20)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tools.plots import plot_train_test_y
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def main():
    window_size = 10
    test_ratio = 0.3

    normalise_window = False
    normalise_y = True

    data = pd.read_csv('data/nasdaq100_df.csv',index_col = 'time')
    data_processing = DataForModel(data, normalise_y, test_ratio)
    start = timer()

    #split data
    X_train, Y_train = data_processing.get_train_batch(window_size, normalise_window)
    # reshape window dimension into features, to fit into xgboost model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))

    # test set
    X_test, Y_test = data_processing.get_test_batch(window_size, normalise_window)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    time_cost = timer() - start
    print(' processing train and test data by batch method took %f s' %time_cost)

    y_pred_train, y_pred_test = svr_predict(X_train,Y_train,X_test,Y_test)

def svr_predict(X_train,Y_train,X_test,Y_test):
    scaler = StandardScaler().fit(X_train)
    x_train_scaled = X_train
    x_test_scaled = X_test
    start = timer()
    model = Ridge(alpha=0.01)
    model.fit(x_train_scaled, Y_train.ravel())
    y_pred_train = model.predict(x_train_scaled)
    y_pred_test = model.predict(x_test_scaled)
    print('the training of linear svr model took {}s.'.format(timer() - start))
    print('explained_variance_score:', explained_variance_score(Y_test, y_pred_test))
    print('final mse: ', mean_squared_error(Y_test, y_pred_test))

    y_true = np.concatenate((Y_train, Y_test))
    plot_train_test_y(False, y_true, y_pred_train, y_pred_test, 'ridge/ridge')

if __name__ =='__main__':
    main()